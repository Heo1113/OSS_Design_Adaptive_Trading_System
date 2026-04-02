import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import random
from multiprocessing import Pool, cpu_count
import os
from numba import njit

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# [1. 설정]
# ─────────────────────────────────────────────
client = Client("", "", {"verify": True, "timeout": 20})
SYMBOL       = 'FARTCOINUSDT'
TOTAL_DAYS   = 540
BUFFER_DAYS  = 60
LEVERAGE     = 5
TAKER_FEE    = 0.0005
SLIPPAGE     = 0.005
BAL_CAP      = 10_000.0

TRAIN_DAYS        = 90
TEST_DAYS_PER_WIN = 30
STEP_DAYS         = 20
PATIENCE          = 40
POP_SIZE          = 8000
GENERATIONS       = 150
MUTATION_RATE     = 0.15
ELITE_SIZE        = 200
MDD_LIMIT         = 0.33

# ─── 레짐 설정 ────────────────────────────────
R_INTER_FIXED    = '1h'
REGIME_SMA_SHORT = 960
REGIME_KAMA_LEN  = 10

# ─── 롤링 볼륨 윈도우 (3m 봉 기준 환산) ─────────
TF_WINDOW = {'15m': 5, '1h': 20, '2h': 40, '4h': 80}

# ─── 클러스터링 설정 ───────────────────────────
TOP_K_COLLECT   = 30
MIN_CLUSTER_WIN = 3
MAX_CLUSTERS    = 8

# ─── OOS 채택 기준 ─────────────────────────────
MIN_TEST_ROI    =  3.0    # v07: 5→3% 완화 (거래수 부족 문제 대응)
MAX_TEST_MDD    =  0.35
MIN_TEST_TRADES =  5      # v07: 10→5 완화 (OOS 거래수 부족 대응)

# ─── 내부 IS/OOS 분리 비율 (안티-오버피팅) ────────
IS_SPLIT_RATIO  = 0.70    # v07 신규: 학습 데이터 70% IS / 30% 내부검증

# ─────────────────────────────────────────────
# [2. 파라미터 범위]
#
# ★ v07 변경점:
#   ① entry_tf 유전자: 1m vs 3m RSI/MFI 진입 신호 선택
#   ② regime_thresh 유전자: 기존 하드코딩 1.0 → 탐색 가능
#   ③ 횡보모드 진입: RSI AND MFI → RSI OR MFI (진입 빈도 증가)
#   ④ Fitness 캡: ROI 80→40, Calmar 10→5 (오버피팅 억제)
#   ⑤ 내부 IS/OOS 분리 Fitness (하모닉 평균): 과최적화 차단
#   ⑥ Range 모드 SL 캡: 2%→5% (FARTCOIN 변동성 대응)
#   ⑦ cooldown_max: 5→3 (연속 손실 후 재진입 빠르게)
#   ⑧ _parse_result 최소 거래수: 10/20→3/8
# ─────────────────────────────────────────────
GENE_BOUNDS = {
    # ── 횡보 모드 (mode 0) ──────────────────────
    'r_adx_limit':    (10.0, 40.0),
    'r_slope_max':    (-10.0, 0.0),
    'r_chop_min':     (40.0, 70.0),
    'r_vol_limit':    (0.1,  3.5),
    'r_chop_15m_min': (40.0, 70.0),
    'r_adx_15m_max':  (10.0, 35.0),

    # [1m/3m 진입 신호] — OR 조건 (AND → OR, v07 변경)
    'rsi_low':        (15.0, 45.0),
    'rsi_high':       (55.0, 85.0),
    'r_mfi_low':      (10.0, 45.0),
    'r_mfi_high':     (55.0, 90.0),

    # TP/SL
    'r_tp_mult':      (0.1,  5.0),
    'r_sl_mult':      (0.1,  4.0),

    # ── 추세 모드 (mode 1/2) ────────────────────
    't_adx_4h_min':   (15.0, 45.0),
    't_chop_4h_max':  (30.0, 65.0),
    't_slope_4h_min': (0.0,  5.0),

    't_macd_1h_thresh': (-0.01, 0.03),

    't_vol_15m_ratio':  (0.1, 4.0),
    't_adx_15m_min':    (10.0, 40.0),

    't_rsi_max':      (50.0, 95.0),
    't_rsi_min':      (5.0,  50.0),
    't_mfi_max':      (50.0, 95.0),
    't_mfi_min':      (5.0,  50.0),

    't_adx_strong_thresh': (25.0, 55.0),

    't_tp_normal_mult': (1.0, 6.0),
    't_tp_strong_mult': (2.0, 10.0),
    't_sl_normal':      (0.5, 5.0),
    't_sl_strong':      (1.0, 6.0),

    # ── 공통 청산 ───────────────────────────────
    't_ts_mult':         (0.002, 0.03),
    't_sl_activate':     (0.01,  0.1),
    't_vol_short_ratio': (0.1,   2.0),
    't_short_vol_exit':  (1.5,   8.0),

    # ── v07 신규: 레짐 임계값 유전자 ───────────────
    # 기존 하드코딩 1.0 → GA가 탐색하도록 변경
    # 1.0 이하: 강세(상승추세), 이상: 약세
    'regime_thresh':     (0.85, 1.15),
}

# v07: entry_tf 추가 (1m vs 3m RSI/MFI 진입 신호 선택)
TF_KEYS        = ['atr_inter', 'entry_tf']
TF_OPTIONS     = ['1h', '2h', '4h']          # atr_inter 옵션
TF_OPTIONS_ENT = ['1m', '3m']                # entry_tf 옵션

_NUMERIC_KEYS = list(GENE_BOUNDS.keys())
_BOUNDS_LO    = np.array([GENE_BOUNDS[k][0] for k in _NUMERIC_KEYS])
_BOUNDS_HI    = np.array([GENE_BOUNDS[k][1] for k in _NUMERIC_KEYS])


# ─────────────────────────────────────────────
# [3. 데이터 로딩]
# ─────────────────────────────────────────────
def get_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    try:
        klines = client.futures_historical_klines(symbol, interval, f"{days} days ago UTC")
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            'ts','open','high','low','close','vol',
            'close_time','qav','trades','tbbav','tbqav','ignore'
        ])
        df = df[['ts','open','high','low','close','vol']].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"  ⚠️ 데이터 오류 ({symbol} {interval}): {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# [4. 데이터 준비]
#
# ★ v07 변경점:
#   - rsi_3m / mfi_3m 보존 (entry_tf='3m' 사용 시)
#   - rsi_1m / mfi_1m (entry_tf='1m' 사용 시)
#   - 나머지 지표 구조는 v06 동일
# ─────────────────────────────────────────────
def prepare_full_data() -> pd.DataFrame | None:
    print(f"🔄 {SYMBOL} 데이터 로딩 중...")
    df_raw = get_data(SYMBOL, '3m', TOTAL_DAYS + BUFFER_DAYS)
    if df_raw.empty:
        print("❌ 3분봉 데이터 로딩 실패")
        return None

    # ── 3m 기준봉 ───────────────────────────────
    df_raw['vol_mean'] = df_raw['vol'].rolling(20).mean()
    df_raw['rsi_3m']   = ta.rsi(df_raw['close'], length=14)
    df_raw['mfi_3m']   = ta.mfi(df_raw['high'], df_raw['low'], df_raw['close'],
                                 df_raw['vol'], length=14)

    # ── 레짐 ────────────────────────────────────
    df_raw['kama_fast']  = ta.kama(df_raw['close'], length=REGIME_KAMA_LEN)
    df_raw['sma_short']  = ta.sma(df_raw['close'],  length=REGIME_SMA_SHORT)
    df_raw['regime_str'] = (
        df_raw['close'] / (df_raw['kama_fast'] + 1e-9) +
        df_raw['close'] / (df_raw['sma_short']  + 1e-9)
    ) / 2.0

    # ── 고정 TF 지표 (1h / 2h / 4h) ─────────────
    for tf in TF_OPTIONS:
        df_tf = get_data(SYMBOL, tf, TOTAL_DAYS + BUFFER_DAYS)
        if df_tf.empty:
            print(f"  ⚠️ {tf} 데이터 로딩 실패, 스킵")
            continue

        df_tf[f'ma20_{tf}']            = ta.sma(df_tf['close'], length=20)
        adx_series                      = ta.adx(df_tf['high'], df_tf['low'], df_tf['close'])['ADX_14']
        df_tf[f'adx_{tf}']             = ta.ema(adx_series, length=3)
        df_tf[f'adx_slope_{tf}']       = ta.ema(adx_series.pct_change() * 100, length=3)
        df_tf[f'atr_{tf}']             = ta.atr(df_tf['high'], df_tf['low'], df_tf['close'], length=14)
        df_tf[f'rsi_{tf}']             = ta.rsi(df_tf['close'], length=14)
        df_tf[f'vol_{tf}_mean']        = df_tf['vol'].rolling(20).mean()
        df_tf[f'mfi_{tf}']             = ta.mfi(df_tf['high'], df_tf['low'],
                                                 df_tf['close'], df_tf['vol'], length=14)
        df_tf[f'chop_{tf}']            = ta.chop(df_tf['high'], df_tf['low'],
                                                  df_tf['close'], length=14)
        macd_df                         = ta.macd(df_tf['close'], fast=12, slow=26, signal=9)
        df_tf[f'macd_hist_norm_{tf}']  = macd_df['MACDh_12_26_9'] / (df_tf['close'] + 1e-9)

        window_size = TF_WINDOW.get(tf, 20)
        df_raw[f'rolling_vol_{tf}'] = df_raw['vol'].rolling(window_size).sum()

        df_raw = pd.merge_asof(
            df_raw.sort_values('ts'),
            df_tf.sort_values('ts'),
            on='ts', direction='backward',
            suffixes=('', f'_{tf}_tmp')
        )

    # ── 15m 데이터 ──────────────────────────────
    df_15m = get_data(SYMBOL, '15m', TOTAL_DAYS + BUFFER_DAYS)
    if not df_15m.empty:
        df_15m['ma20_15m']     = ta.sma(df_15m['close'], length=20)
        adx_15m_series          = ta.adx(df_15m['high'], df_15m['low'], df_15m['close'])['ADX_14']
        df_15m['adx_15m']      = ta.ema(adx_15m_series, length=3)
        df_15m['chop_15m']     = ta.chop(df_15m['high'], df_15m['low'],
                                          df_15m['close'], length=14)
        df_15m['vol_15m_mean'] = df_15m['vol'].rolling(20).mean()

        df_raw['rolling_vol_15m'] = df_raw['vol'].rolling(TF_WINDOW['15m']).sum()

        df_raw = pd.merge_asof(
            df_raw.sort_values('ts'),
            df_15m[['ts', 'ma20_15m', 'adx_15m', 'chop_15m', 'vol_15m_mean']].sort_values('ts'),
            on='ts', direction='backward'
        )
        print("  ✅ 15m 데이터 로딩 완료")
    else:
        df_raw['ma20_15m']        = df_raw[f'ma20_{R_INTER_FIXED}']
        df_raw['adx_15m']         = df_raw[f'adx_{R_INTER_FIXED}']
        df_raw['chop_15m']        = df_raw[f'chop_{R_INTER_FIXED}']
        df_raw['vol_15m_mean']    = df_raw['vol_mean']
        df_raw['rolling_vol_15m'] = df_raw[f'rolling_vol_{R_INTER_FIXED}']
        print("  ⚠️ 15m 데이터 로딩 실패 — 1h 지표로 대체")

    # ── 1m 데이터 ───────────────────────────────
    df_1m = get_data(SYMBOL, '1m', TOTAL_DAYS + BUFFER_DAYS)
    if not df_1m.empty:
        df_1m['rsi_1m'] = ta.rsi(df_1m['close'], length=14)
        df_1m['mfi_1m'] = ta.mfi(df_1m['high'], df_1m['low'],
                                  df_1m['close'], df_1m['vol'], length=14)
        df_raw = pd.merge_asof(
            df_raw.sort_values('ts'),
            df_1m[['ts', 'rsi_1m', 'mfi_1m']].sort_values('ts'),
            on='ts', direction='backward'
        )
        print("  ✅ 1m 데이터 로딩 완료")
    else:
        df_raw['rsi_1m'] = df_raw['rsi_3m']
        df_raw['mfi_1m'] = df_raw['mfi_3m']
        print("  ⚠️ 1m 데이터 로딩 실패 — 3m 지표로 대체")

    return df_raw.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────
# [5. Numba 백테스트 코어]
#
# ★ v07 변경점:
#   ① rsi_3m / mfi_3m 추가 파라미터
#   ② use_1m 플래그: 1.0이면 1m 신호 사용, 0.0이면 3m 신호 사용
#   ③ regime_thresh 파라미터 추가 (p_vals 마지막)
#   ④ 횡보 진입: RSI AND MFI → RSI OR MFI
#      (둘 중 하나만 충족해도 진입 → 거래 빈도 증가)
#   ⑤ cooldown_max: 5→3
#   ⑥ Range 모드 SL 캡: 0.02→0.05
# ─────────────────────────────────────────────
@njit(cache=True)
def _run_backtest_core(
    # ── 3m 기준봉 ────────────────────────────────
    close, open_, vol, vol_mean,
    # ── 1m 진입 신호 ─────────────────────────────
    rsi_1m, mfi_1m,
    # ── 3m 진입 신호 (v07 신규) ───────────────────
    rsi_3m, mfi_3m,
    # ── entry_tf 선택 플래그 (v07 신규) ───────────
    # 1.0 = 1m 신호 사용, 0.0 = 3m 신호 사용
    use_1m,
    # ── 15m ──────────────────────────────────────
    ma20_15m, adx_15m, chop_15m, vol_15m_mean, rolling_vol_15m,
    # ── 1h ───────────────────────────────────────
    adx_1h, adx_slope_1h, chop_1h, ma20_1h, macd_hist_1h,
    # ── 4h ───────────────────────────────────────
    adx_4h, adx_slope_4h, chop_4h,
    # ── 공통 ─────────────────────────────────────
    atr,
    regime,
    p_vals, lev, fee, slip, mdd_lim, b_cap
):
    # ── 파라미터 언패킹 (GENE_BOUNDS 순서와 일치) ──
    (r_adx_limit, r_slope_max, r_chop_min, r_vol_limit,
     r_chop_15m_min, r_adx_15m_max,
     rsi_low, rsi_high, r_mfi_low, r_mfi_high,
     r_tp_mult, r_sl_mult,
     t_adx_4h_min, t_chop_4h_max, t_slope_4h_min,
     t_macd_1h_thresh,
     t_vol_15m_ratio, t_adx_15m_min,
     t_rsi_max, t_rsi_min, t_mfi_max, t_mfi_min,
     t_adx_strong_thresh,
     t_tp_normal_mult, t_tp_strong_mult,
     t_sl_normal, t_sl_strong,
     t_ts_mult, t_sl_activate,
     t_vol_short_ratio, t_short_vol_exit,
     regime_thresh) = p_vals             # v07: regime_thresh 추가

    cooldown_max   = 3    # v07: 5→3 (연속손실 후 재진입 빠르게)
    consec_loss_th = 4    # v07: 3→4 (쿨다운 트리거 조금 느리게)

    bal   = 100.0; peak = 100.0; mdd = 0.0
    in_pos = False; side = 0; mode = 0; ent_p = 0.0
    sl = 0.0; tp = 0.0; dur = 0
    ent_regime_scale = 1.0
    tr_counts  = np.zeros(3)
    trade_pnls = []

    consec_loss  = 0
    cooldown     = 0
    RING_SIZE    = 10
    ring_buf     = np.zeros(RING_SIZE)
    ring_head    = 0
    ring_filled  = 0
    ent_bal      = 100.0
    ent_pos_size = 0.0

    for i in range(len(close)):
        cp = close[i]

        # ── v07: entry_tf 기반 신호 선택 ──────────
        if use_1m > 0.5:
            rsi_sig = rsi_1m[i]
            mfi_sig = mfi_1m[i]
        else:
            rsi_sig = rsi_3m[i]
            mfi_sig = mfi_3m[i]

        if not in_pos:
            if cooldown > 0:
                cooldown -= 1
                continue

            new_side = 0; new_mode = -1

            # ─────────────────────────────────────
            # ① 횡보 진입 (mode 0)
            #
            # [1h] ADX 낮음 + 기울기 감소 + Choppiness 높음 + 거래량
            # [15m] Choppiness 높음 + ADX 낮음
            # [신호] RSI OR MFI (v07: AND→OR, 진입빈도 증가)
            #        + 1h/15m MA 방향 필터
            # ─────────────────────────────────────
            if (adx_1h[i]       < r_adx_limit
                    and adx_slope_1h[i] <= r_slope_max
                    and chop_1h[i]      >= r_chop_min
                    and vol[i]          >  vol_mean[i] * r_vol_limit):

                if (chop_15m[i]  >= r_chop_15m_min
                        and adx_15m[i] <  r_adx_15m_max):

                    # 롱: (RSI 과매도 OR MFI 과매도) + MA 하방
                    if ((rsi_sig < rsi_low or mfi_sig < r_mfi_low)
                            and cp < ma20_1h[i]
                            and cp < ma20_15m[i]):
                        new_side =  1; new_mode = 0

                    # 숏: (RSI 과매수 OR MFI 과매수) + MA 상방
                    elif ((rsi_sig > rsi_high or mfi_sig > r_mfi_high)
                            and cp > ma20_1h[i]
                            and cp > ma20_15m[i]):
                        new_side = -1; new_mode = 0

            # ─────────────────────────────────────
            # ② 추세 진입 (mode 1/2)
            #
            # [4h] ADX 높음 + 기울기 양수 + Choppiness 낮음
            # [1h] MACD 방향 + 가격 MA 위치
            # [15m] 거래량 + ADX + 가격 MA
            # [신호] RSI/MFI 오버바이·오버셀 차단 (AND 유지)
            # ─────────────────────────────────────
            elif (adx_4h[i]        >  t_adx_4h_min
                    and adx_slope_4h[i] >= t_slope_4h_min
                    and chop_4h[i]      <  t_chop_4h_max):

                cur_mode = 2 if adx_4h[i] > t_adx_strong_thresh else 1

                if (macd_hist_1h[i]  >  t_macd_1h_thresh
                        and cp          >  ma20_1h[i]):

                    if (rolling_vol_15m[i] > vol_15m_mean[i] * t_vol_15m_ratio
                            and adx_15m[i]     >  t_adx_15m_min
                            and cp             >  ma20_15m[i]):

                        if (rsi_sig < t_rsi_max
                                and mfi_sig < t_mfi_max):
                            new_side =  1; new_mode = cur_mode

                elif (macd_hist_1h[i] < -t_macd_1h_thresh
                        and cp            <  ma20_1h[i]):

                    if (rolling_vol_15m[i] > vol_15m_mean[i] * t_vol_15m_ratio * t_vol_short_ratio
                            and adx_15m[i]     >  t_adx_15m_min
                            and cp             <  ma20_15m[i]):

                        if (rsi_sig > t_rsi_min
                                and mfi_sig > t_mfi_min):
                            new_side = -1; new_mode = cur_mode

            # ─────────────────────────────────────
            # 레짐 필터
            # v07: regime_thresh 유전자로 탐색 (기존 하드코딩 1.0 제거)
            # ─────────────────────────────────────
            if new_mode >= 0:
                counter_trend = (
                    (new_side ==  1 and regime[i] <  regime_thresh) or
                    (new_side == -1 and regime[i] >= regime_thresh)
                )
                if counter_trend:
                    new_mode = -1

            if new_mode >= 0:
                ent_regime_scale = 1.0

                atr_pct = atr[i] / (cp + 1e-9)
                if new_mode == 0:
                    tp_p = atr_pct * r_tp_mult
                    # v07: SL 캡 0.02→0.05 (FARTCOIN 변동성 대응)
                    sl_p = min(atr_pct * r_sl_mult, 0.05)
                elif new_mode == 2:
                    tp_p = atr_pct * t_tp_strong_mult
                    sl_p = min(atr_pct * t_sl_strong, 0.05)
                else:
                    tp_p = atr_pct * t_tp_normal_mult
                    sl_p = min(atr_pct * t_sl_normal, 0.05)

                tp = cp * (1 + tp_p) if new_side == 1 else cp * (1 - tp_p)
                sl = cp * (1 - sl_p) if new_side == 1 else cp * (1 + sl_p)
                side, mode, ent_p, dur, in_pos = new_side, new_mode, cp, 0, True

                pos_rate_cap = min(1.0, b_cap / (bal + 1e-9))
                if ring_filled >= RING_SIZE:
                    rec_gp = 0.0; rec_gl = 1e-9
                    for j in range(RING_SIZE):
                        v = ring_buf[j]
                        if v > 0: rec_gp += v
                        else:     rec_gl += abs(v)
                    pf = rec_gp / rec_gl
                    perf_scale = min(1.0, max(0.5, pf / 1.5))
                else:
                    perf_scale = 0.75

                ent_bal      = bal
                ent_pos_size = ent_bal * pos_rate_cap * perf_scale * ent_regime_scale

        else:
            dur += 1
            is_exit = False
            exit_p  = cp

            # ── 트레일링 스탑 (추세 모드) ────────────
            if mode != 0:
                if side == 1 and (cp - ent_p) / ent_p > t_sl_activate:
                    tsl = cp * (1 - t_ts_mult)
                    if tsl > sl: sl = tsl
                elif side == -1 and (ent_p - cp) / ent_p > t_sl_activate:
                    tsl = cp * (1 + t_ts_mult)
                    if tsl < sl: sl = tsl

            # ── 데드캣 조기 청산 ─────────────────────
            if mode != 0 and side == -1:
                vol_exit_cond = rolling_vol_15m[i] > vol_15m_mean[i] * t_short_vol_exit
                if close[i] > open_[i] and cp > ma20_1h[i] and vol_exit_cond:
                    is_exit = True

            # ── TP / SL / 타임아웃 ───────────────────
            if not is_exit:
                if mode == 0 and dur >= 15:
                    is_exit = True
                elif (side == 1 and cp <= sl) or (side == -1 and cp >= sl):
                    is_exit = True; exit_p = sl
                elif (side == 1 and cp >= tp) or (side == -1 and cp <= tp):
                    is_exit = True; exit_p = tp

            if is_exit:
                raw_ret = ((exit_p - ent_p) / ent_p) if side == 1 else ((ent_p - exit_p) / ent_p)
                pnl     = ent_pos_size * (raw_ret - fee * 2 - slip) * lev
                bal    += pnl
                trade_pnls.append(pnl)

                ring_buf[ring_head] = pnl
                ring_head = (ring_head + 1) % RING_SIZE
                if ring_filled < RING_SIZE: ring_filled += 1

                if bal > peak: peak = bal
                mdd = max(mdd, (peak - bal) / peak)
                tr_counts[mode] += 1
                if pnl > 0:
                    consec_loss = 0
                else:
                    consec_loss += 1
                    if consec_loss >= consec_loss_th:
                        cooldown    = cooldown_max
                        consec_loss = 0
                in_pos = False
                if mdd > mdd_lim or bal <= 5.0:
                    break

    return bal, mdd, tr_counts, np.array(trade_pnls)


# ─────────────────────────────────────────────
# [6. 평가 인프라]
#
# ★ v07 변경점:
#   ① _init_worker: df_is / df_ov 두 개 분리 저장
#   ② evaluate_fast: IS + OOS_val 각각 백테스트 후 하모닉 평균
#      → 한쪽에만 과최적화된 파라미터 자동 탈락
#   ③ _parse_result: 최소 거래수 10/20→3/8
#   ④ Fitness ROI 캡 80→40, Calmar 캡 10→5
# ─────────────────────────────────────────────
_worker_data: dict = {}

def _init_worker(df_is: pd.DataFrame, df_ov: pd.DataFrame) -> None:
    """v07: IS / OOS_val 두 개 데이터셋 공유"""
    global _worker_data
    _worker_data['df_is'] = df_is
    _worker_data['df_ov'] = df_ov


def _build_core_args(ind: dict, df: pd.DataFrame):
    """
    ★ v07 변경점:
      - rsi_3m / mfi_3m 추가 (entry_tf='3m' 선택 시 사용)
      - use_1m 스칼라 플래그 추가
      - regime_thresh p_vals 마지막에 포함 (GENE_BOUNDS에 있어 자동 포함)
    """
    atr_tf = ind['atr_inter']
    use_1m = 1.0 if ind.get('entry_tf', '1m') == '1m' else 0.0
    p_vals = np.array([ind[k] for k in _NUMERIC_KEYS], dtype=np.float64)

    r_tf = R_INTER_FIXED  # '1h'

    return (
        df['close'].values,
        df['open'].values,
        df['vol'].values,
        df['vol_mean'].values,
        # 1m 신호
        df['rsi_1m'].values,
        df['mfi_1m'].values,
        # 3m 신호 (v07 신규)
        df['rsi_3m'].values,
        df['mfi_3m'].values,
        # entry_tf 플래그 (v07 신규)
        use_1m,
        # 15m
        df['ma20_15m'].values,
        df['adx_15m'].values,
        df['chop_15m'].values,
        df['vol_15m_mean'].values,
        df['rolling_vol_15m'].values,
        # 1h
        df[f'adx_{r_tf}'].values,
        df[f'adx_slope_{r_tf}'].values,
        df[f'chop_{r_tf}'].values,
        df[f'ma20_{r_tf}'].values,
        df[f'macd_hist_norm_{r_tf}'].values,
        # 4h
        df['adx_4h'].values,
        df['adx_slope_4h'].values,
        df['chop_4h'].values,
        # ATR
        df[f'atr_{atr_tf}'].values,
        # 레짐
        df['regime_str'].values,
        p_vals, LEVERAGE, TAKER_FEE, SLIPPAGE, MDD_LIMIT, BAL_CAP
    )


def _parse_result(ind: dict, bal, mdd, tr_c, pnls) -> dict:
    total_tr = int(np.sum(tr_c))
    roi      = bal - 100.0
    # v07: 최소 거래수 10/20→3/8 (OOS 거래 부족 문제 대응)
    if total_tr < 3 or mdd > MDD_LIMIT or bal <= 5.0:
        return {**ind, 'Fitness': -1e6, 'ROI': roi, 'MDD': mdd, 'Trades': total_tr}
    if total_tr < 8 or bal <= 10.0:
        return {**ind, 'Fitness': -1e7, 'ROI': roi, 'MDD': mdd, 'Trades': total_tr}
    calmar      = roi / (mdd * 100 + 1e-9)
    consistency = (pnls.mean() / (pnls.std() + 1e-9)) if len(pnls) > 0 else 0.0
    # v07: ROI 캡 80→40, Calmar 캡 10→5 (과최적화 억제 강화)
    roi_capped    = min(roi, 40.0)
    calmar_capped = min(calmar, 5.0)
    fitness       = np.log1p(max(0.0, roi_capped)) * calmar_capped * consistency * np.log10(total_tr + 1)
    return {**ind, 'Fitness': fitness, 'ROI': roi, 'MDD': mdd, 'Trades': total_tr}


def evaluate_fast(ind: dict) -> dict:
    """
    v07: IS + OOS_val 분리 평가 + 하모닉 평균 Fitness
    두 기간 모두에서 수익이 나야 높은 점수 → 과최적화 자동 탈락
    """
    df_is = _worker_data['df_is']
    df_ov = _worker_data['df_ov']
    try:
        res_is = _run_backtest_core(*_build_core_args(ind, df_is))
        res_ov = _run_backtest_core(*_build_core_args(ind, df_ov))
    except Exception:
        return {**ind, 'Fitness': -1e9, 'ROI': -100.0, 'MDD': 1.0, 'Trades': 0}

    r_is = _parse_result(ind, *res_is)
    r_ov = _parse_result(ind, *res_ov)

    f_is = r_is['Fitness']
    f_ov = r_ov['Fitness']

    # 하모닉 평균: 한쪽만 잘되면 낮은 점수 → 두 기간 모두 안정적인 파라미터 선택
    if f_is <= 0 or f_ov <= 0:
        combined = min(f_is, f_ov)
    else:
        combined = 2.0 * f_is * f_ov / (f_is + f_ov)

    return {**ind, 'Fitness': combined, 'ROI': r_is['ROI'],
            'MDD': r_is['MDD'], 'Trades': r_is['Trades']}


def evaluate_on_df(ind: dict, df: pd.DataFrame) -> dict:
    """테스트/클러스터 검증 전용"""
    try:
        res = _run_backtest_core(*_build_core_args(ind, df))
    except Exception:
        return {**ind, 'Fitness': -1e9, 'ROI': -100.0, 'MDD': 1.0, 'Trades': 0}
    return _parse_result(ind, *res)


# ─────────────────────────────────────────────
# [7. 유전 알고리즘]
#
# ★ v07 변경점:
#   ① 학습 데이터 IS/OOS_val 분리 후 _init_worker에 전달
#   ② entry_tf 유전자 교배/돌연변이 처리
#   ③ IS 분리 비율: IS_SPLIT_RATIO (기본 0.70)
# ─────────────────────────────────────────────
def run_ga(df_train: pd.DataFrame) -> tuple[dict | None, list[dict]]:
    gene_keys = list(_NUMERIC_KEYS) + TF_KEYS

    # v07: IS / OOS_val 분리 (안티-오버피팅 핵심)
    n = len(df_train)
    split_idx = int(n * IS_SPLIT_RATIO)
    df_is = df_train.iloc[:split_idx].reset_index(drop=True)
    df_ov = df_train.iloc[split_idx:].reset_index(drop=True)
    print(f"   IS: {len(df_is):,}봉 / OOS_val: {len(df_ov):,}봉 (분리비율 {IS_SPLIT_RATIO:.0%})")

    population = [
        {
            **{k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()},
            'atr_inter': random.choice(TF_OPTIONS),
            'entry_tf':  random.choice(TF_OPTIONS_ENT),   # v07: entry_tf 추가
        }
        for _ in range(POP_SIZE)
    ]
    best_overall_fitness = -float('inf')
    no_improvement_count = 0
    best_individual      = None
    final_elites         = []

    with Pool(cpu_count(), initializer=_init_worker, initargs=(df_is, df_ov)) as pool:
        for gen in range(GENERATIONS):
            results = pool.map(evaluate_fast, population)
            results.sort(key=lambda x: x['Fitness'], reverse=True)

            if results[0]['Fitness'] > best_overall_fitness:
                best_overall_fitness = results[0]['Fitness']
                best_individual      = results[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= PATIENCE:
                print(f"   ⏱️ {PATIENCE}세대 정체 → 조기 종료 (세대: {gen + 1})")
                break

            elites       = results[:ELITE_SIZE]
            final_elites = elites
            new_pop = [
                {k: v for k, v in e.items() if k in gene_keys}
                for e in elites
            ]
            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(elites, 2)
                child = {
                    k: (random.choice([p1[k], p2[k]])
                        if random.random() > MUTATION_RATE
                        else random.uniform(GENE_BOUNDS[k][0], GENE_BOUNDS[k][1]))
                    for k in _NUMERIC_KEYS
                }
                child['atr_inter'] = random.choice([p1['atr_inter'], p2['atr_inter']])
                # v07: entry_tf 교배/돌연변이
                child['entry_tf']  = (
                    random.choice([p1['entry_tf'], p2['entry_tf']])
                    if random.random() > MUTATION_RATE
                    else random.choice(TF_OPTIONS_ENT)
                )
                new_pop.append(child)
            population = new_pop

    top_k = [
        {k: v for k, v in e.items() if k in gene_keys}
        for e in final_elites[:TOP_K_COLLECT]
    ]
    return best_individual, top_k


# ─────────────────────────────────────────────
# [8. 클러스터링 분석]
# ─────────────────────────────────────────────
def _normalize(candidates: list[dict]) -> np.ndarray:
    X = np.array([[c[k] for k in _NUMERIC_KEYS] for c in candidates])
    return (X - _BOUNDS_LO) / (_BOUNDS_HI - _BOUNDS_LO + 1e-9)


def _find_optimal_k(X_norm: np.ndarray, k_range: range) -> int:
    best_k, best_score = k_range.start, -1.0
    for k in k_range:
        if k >= len(X_norm):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_norm)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_norm, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _build_medoids(
    candidates: list[dict],
    X_norm: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> list[dict]:
    medoids = []
    for c in range(n_clusters):
        mask      = labels == c
        members_X = X_norm[mask]
        members_c = [cand for cand, m in zip(candidates, mask) if m]
        if not members_c:
            continue

        centroid  = members_X.mean(axis=0)
        dists     = np.linalg.norm(members_X - centroid, axis=1)
        best_idx  = int(np.argmin(dists))
        medoid    = members_c[best_idx].copy()

        # TF 유전자: 클러스터 내 다수결
        for tf_key in TF_KEYS:
            vals = [m[tf_key] for m in members_c]
            medoid[tf_key] = max(set(vals), key=vals.count)

        medoids.append({
            'cluster_id'   : c,
            'cluster_size' : int(mask.sum()),
            'centroid_dist': round(float(dists[best_idx]), 6),
            'medoid'       : medoid,
        })
    return medoids


def run_cluster_analysis(
    all_candidates: list[dict],
    test_windows: list[tuple],
    file_cluster_csv: str,
    file_best_params_csv: str,
) -> dict | None:
    n_cands = len(all_candidates)
    n_wins  = len(test_windows)
    print(f"\n{'='*60}")
    print(f"🔬 클러스터 분석 시작 | 후보: {n_cands}개 / 테스트 윈도우: {n_wins}개")

    if n_wins < MIN_CLUSTER_WIN:
        print(f"   ⚠️ 테스트 윈도우 {n_wins}개 < 최소 {MIN_CLUSTER_WIN}개 → 스킵")
        return None

    X_norm = _normalize(all_candidates)
    k_max  = min(MAX_CLUSTERS, n_wins, n_cands // 2)
    if k_max < 2:
        print("   ⚠️ 클러스터 수 부족 → 스킵")
        return None

    best_k  = _find_optimal_k(X_norm, range(2, k_max + 1))
    labels  = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(X_norm)
    medoids = _build_medoids(all_candidates, X_norm, labels, best_k)
    print(f"   최적 K: {best_k} (후보 {n_cands}개 → {len(medoids)}개 클러스터)")

    cluster_rows = []
    best_medoid  = None
    best_calmar  = -float('inf')

    for info in medoids:
        med    = info['medoid']
        rois   = []
        mdds_w = []
        trades = []

        for win_idx, t_start, t_end, te_end, df_test in test_windows:
            r = evaluate_on_df(med, df_test)
            rois.append(r['ROI'])
            mdds_w.append(r['MDD'])
            trades.append(r['Trades'])

        valid_rois = [roi for roi, t in zip(rois, trades) if t >= MIN_TEST_TRADES]
        avg_roi    = float(np.mean(valid_rois)) if valid_rois else -999.0
        win_rate   = sum(1 for roi in valid_rois if roi >= MIN_TEST_ROI) / max(len(valid_rois), 1)
        max_mdd    = float(max(mdds_w)) if mdds_w else 1.0
        calmar     = avg_roi / (max_mdd * 100 + 1e-9) if valid_rois else -999.0

        c_bal = 100.0; c_peak = 100.0; c_mdd = 0.0
        for roi in rois:
            c_bal *= max(1.0 + min(roi, BAL_CAP - 100.0) / 100.0, 0.0)
            if c_bal > c_peak: c_peak = c_bal
            c_mdd = max(c_mdd, (c_peak - c_bal) / (c_peak + 1e-9))

        row = {
            'cluster_id'    : info['cluster_id'],
            'cluster_size'  : info['cluster_size'],
            'centroid_dist' : info['centroid_dist'],
            'avg_oos_roi'   : round(avg_roi, 4),
            'oos_win_rate'  : round(win_rate, 4),
            'max_oos_mdd'   : round(max_mdd, 4),
            'calmar'        : round(calmar, 4),
            'oos_final_bal' : round(c_bal, 4),
            'oos_cum_mdd'   : round(c_mdd, 4),
            'window_rois'   : str([round(r, 2) for r in rois]),
        }
        cluster_rows.append(row)

        print(f"   클러스터 {info['cluster_id']:2d} "
              f"(크기:{info['cluster_size']:3d}) | "
              f"avg ROI: {avg_roi:+6.2f}% | 승률: {win_rate*100:.0f}% | "
              f"MDD: {max_mdd*100:.1f}% | Calmar: {calmar:.3f} | OOS잔고: ${c_bal:,.2f}")

        if calmar > best_calmar:
            best_calmar = calmar
            best_medoid = med.copy()
            best_medoid['cluster_id']    = info['cluster_id']
            best_medoid['cluster_size']  = info['cluster_size']
            best_medoid['avg_oos_roi']   = round(avg_roi, 4)
            best_medoid['oos_win_rate']  = round(win_rate, 4)
            best_medoid['max_oos_mdd']   = round(max_mdd, 4)
            best_medoid['calmar']        = round(calmar, 4)
            best_medoid['oos_final_bal'] = round(c_bal, 4)

    pd.DataFrame(cluster_rows).to_csv(file_cluster_csv, index=False, encoding='utf-8-sig')
    print(f"\n   💾 클러스터 분석 결과 → {file_cluster_csv}")

    if best_medoid is not None:
        print(f"\n   🏆 최우수 클러스터: {best_medoid['cluster_id']} "
              f"| Calmar: {best_medoid['calmar']:.3f} "
              f"| avg ROI: {best_medoid['avg_oos_roi']:+.2f}% "
              f"| OOS 잔고: ${best_medoid['oos_final_bal']:,.2f}")
        pd.DataFrame([best_medoid]).to_csv(file_best_params_csv, index=False, encoding='utf-8-sig')
        print(f"   💾 최우수 클러스터 파라미터 → {file_best_params_csv}")

    return best_medoid


# ─────────────────────────────────────────────
# [9. WFA 메인 루프]
# ─────────────────────────────────────────────
if __name__ == "__main__":
    FILE_OOS          = "WFA_OOS_Summary.csv"
    FILE_PARAMS       = "WFA_Params.csv"
    FILE_CLUSTER      = "WFA_Cluster_Analysis.csv"
    FILE_CLUSTER_BEST = "WFA_Cluster_Best_Params.csv"

    df_all = prepare_full_data()
    if df_all is None:
        exit(1)

    if os.path.exists(FILE_OOS):
        prev                = pd.read_csv(FILE_OOS)
        win_idx             = int(prev['window'].max())
        oos_bal             = float(prev['oos_bal'].iloc[-1])
        oos_peak            = float(prev['oos_bal'].max())
        oos_mdd             = float(prev['oos_mdd_cumul'].iloc[-1]) if 'oos_mdd_cumul' in prev.columns else 0.0
        last_test_end       = pd.to_datetime(prev['test_end'].iloc[-1])
        current_train_start = last_test_end - pd.Timedelta(days=TEST_DAYS_PER_WIN + TRAIN_DAYS - STEP_DAYS)
        print(f"⏩ 기존 파일 발견 → 윈도우 {win_idx}부터 이어서 시작")
    else:
        win_idx             = 0
        oos_bal             = 100.0
        oos_peak            = 100.0
        oos_mdd             = 0.0
        current_train_start = df_all['ts'].min()
        print("🆕 새로 시작")

    all_candidates: list[dict] = []
    test_windows:   list[tuple] = []

    # ─── Phase 1: 모든 윈도우 GA 학습 + 후보 수집 ─
    while True:
        train_end = current_train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_end  = train_end           + pd.Timedelta(days=TEST_DAYS_PER_WIN)
        if test_end > df_all['ts'].max():
            break

        win_idx += 1
        print(f"\n{'='*60}")
        print(f"📂 윈도우 {win_idx} | 학습: {current_train_start.date()} ~ {train_end.date()} "
              f"| 테스트: {train_end.date()} ~ {test_end.date()}")

        df_train           = df_all[(df_all['ts'] >= current_train_start) & (df_all['ts'] < train_end)].copy()
        best_params, top_k = run_ga(df_train)

        if not best_params or best_params.get('Fitness', -1e9) <= -1000:
            print("   ⚠️ 유효한 파라미터 없음 → 건너뜀")
            current_train_start += pd.Timedelta(days=STEP_DAYS)
            continue

        all_candidates.extend(top_k)

        train_roi = best_params['ROI']
        print(f"   [학습-IS] ROI: {train_roi:.2f}% | "
              f"MDD: {best_params['MDD'] * 100:.1f}% | Trades: {best_params['Trades']} "
              f"| 클러스터 풀: {len(all_candidates)}개 "
              f"| entry_tf: {best_params.get('entry_tf', 'N/A')}")

        df_test     = df_all[(df_all['ts'] >= train_end) & (df_all['ts'] < test_end)].copy()
        test_result = evaluate_on_df(best_params, df_test)
        test_roi    = test_result['ROI']
        test_mdd    = test_result['MDD']
        test_trades = test_result['Trades']

        test_windows.append((win_idx, current_train_start, train_end, test_end, df_test))

        reason = []
        if test_roi    < MIN_TEST_ROI:    reason.append(f"ROI {test_roi:.1f}% < {MIN_TEST_ROI}%")
        if test_mdd    > MAX_TEST_MDD:    reason.append(f"MDD {test_mdd * 100:.1f}% > {MAX_TEST_MDD * 100:.0f}%")
        if test_trades < MIN_TEST_TRADES: reason.append(f"거래 {test_trades}회 < {MIN_TEST_TRADES}회")

        is_accepted = len(reason) == 0
        status      = "✅ 채택" if is_accepted else f"❌ 기각 ({' / '.join(reason)})"
        print(f"   [테스트] ROI: {test_roi:.2f}% | MDD: {test_mdd * 100:.1f}% | "
              f"Trades: {test_trades} → {status}")

        if test_trades > 0:
            capped_roi = min(test_roi, BAL_CAP - 100.0)
            oos_bal   *= max(1.0 + capped_roi / 100.0, 0.0)
            if oos_bal > oos_peak: oos_peak = oos_bal
            oos_mdd = max(oos_mdd, (oos_peak - oos_bal) / (oos_peak + 1e-9))

        oos_row = pd.DataFrame([{
            'window'       : win_idx,
            'train_start'  : current_train_start.date(),
            'train_end'    : train_end.date(),
            'test_end'     : test_end.date(),
            'train_roi'    : round(train_roi, 4),
            'test_roi'     : round(test_roi, 4),
            'test_mdd'     : round(test_mdd, 4),
            'test_trades'  : test_trades,
            'is_accepted'  : is_accepted,
            'oos_bal'      : round(oos_bal, 4),
            'oos_mdd_cumul': round(oos_mdd, 4),
            'entry_tf'     : best_params.get('entry_tf', 'N/A'),  # v07: entry_tf 기록
        }])
        oos_row.to_csv(FILE_OOS, mode='a',
                       header=not os.path.exists(FILE_OOS) or win_idx == 1,
                       index=False, encoding='utf-8-sig')

        if is_accepted:
            param_row = best_params.copy()
            param_row.update({
                'win_idx'   : win_idx,
                'train_roi' : round(train_roi, 4),
                'test_roi'  : round(test_roi, 4),
                'test_mdd'  : round(test_mdd, 4),
                'start_date': current_train_start.date(),
                'end_date'  : train_end.date(),
            })
            pd.DataFrame([param_row]).to_csv(
                FILE_PARAMS, mode='a',
                header=not os.path.exists(FILE_PARAMS),
                index=False, encoding='utf-8-sig'
            )

        print(f"   💾 저장 → {FILE_OOS}" + (f" / {FILE_PARAMS}" if is_accepted else ""))
        current_train_start += pd.Timedelta(days=STEP_DAYS)

    # ─── Phase 2: 클러스터링 + OOS 재검증 ──────────
    best_cluster_params = run_cluster_analysis(
        all_candidates, test_windows,
        FILE_CLUSTER, FILE_CLUSTER_BEST,
    )

    # ─── 최종 요약 ─────────────────────────────────
    print(f"\n{'='*60}")
    print("🏁 WFA v07 완료")

    if os.path.exists(FILE_OOS):
        df_oos     = pd.read_csv(FILE_OOS)
        total_wins = int(df_oos['window'].max())
        accepted   = int(df_oos['is_accepted'].sum())
        print(f"\n[전통 WFA] 총 {total_wins}개 윈도우 | 채택: {accepted}개 "
              f"({accepted / total_wins * 100:.1f}%)")
        print(f"   최종 OOS 잔고: ${df_oos['oos_bal'].iloc[-1]:,.2f}  (시작: $100.00)")
        print(f"   OOS 총 수익률: {df_oos['oos_bal'].iloc[-1] - 100:+.2f}%")
        print(f"   OOS 최대 낙폭: {df_oos['oos_mdd_cumul'].max() * 100:.2f}%")

        # entry_tf 선택 통계
        if 'entry_tf' in df_oos.columns:
            tf_counts = df_oos['entry_tf'].value_counts()
            print(f"\n   [entry_tf 선택] {tf_counts.to_dict()}")

        print(f"\n{'─'*60}")
        header = f"{'Win':>4} | {'Train ROI':>9} | {'Test ROI':>8} | {'MDD':>6} | {'채택':>4} | {'TF':>3} | OOS잔고"
        print(header)
        print("─" * 65)
        for _, r in df_oos.iterrows():
            flag = "✅" if r['is_accepted'] else "❌"
            tf_str = str(r.get('entry_tf', '?'))[:3]
            print(f"  {int(r['window']):2d}  | {r['train_roi']:+8.2f}%  | {r['test_roi']:+7.2f}%  | "
                  f"{r['test_mdd'] * 100:5.1f}% | {flag}   | {tf_str} | ${r['oos_bal']:,.2f}")

    if best_cluster_params is not None:
        print(f"\n[클러스터 WFA] 최우수 파라미터 세트")
        print(f"   클러스터 {best_cluster_params['cluster_id']} "
              f"(크기: {best_cluster_params['cluster_size']}개 후보)")
        print(f"   전체 윈도우 평균 OOS ROI: {best_cluster_params['avg_oos_roi']:+.2f}%")
        print(f"   OOS 승률: {best_cluster_params['oos_win_rate'] * 100:.0f}%")
        print(f"   최대 OOS MDD: {best_cluster_params['max_oos_mdd'] * 100:.1f}%")
        print(f"   Calmar: {best_cluster_params['calmar']:.3f}")
        print(f"   누적 OOS 잔고: ${best_cluster_params['oos_final_bal']:,.2f}")
        print(f"   entry_tf: {best_cluster_params.get('entry_tf', 'N/A')}")
        print(f"   regime_thresh: {best_cluster_params.get('regime_thresh', 'N/A'):.4f}")

    print(f"\n💾 파일 목록")
    for f in [FILE_OOS, FILE_PARAMS, FILE_CLUSTER, FILE_CLUSTER_BEST]:
        if os.path.exists(f):
            print(f"   {f}")
    print("=" * 60)
