import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import warnings
import random
from multiprocessing import Pool, cpu_count
import os
import time

warnings.filterwarnings('ignore')

# [1. 설정 영역]
client = Client("", "", {"verify": True, "timeout": 20}) 
SYMBOL = 'FARTCOINUSDT'
TOTAL_DAYS = 480      
BUFFER_DAYS = 60      
LEVERAGE = 5          

TAKER_FEE = 0.0005
SLIPPAGE = 0.001

# 잔고 상한선: 이 값 이상으로 복리 증가를 허용하지 않음
# 실전에서는 일정 수익 이후 포지션 크기를 고정하는 것과 동일한 효과
# 100x 수익(10000%)을 넘으면 현실적으로 의미 없는 과최적화로 판단
BAL_CAP = 10_000.0

# WFA 전용 설정
TRAIN_DAYS = 30     
TEST_DAYS_PER_WIN = 15 
STEP_DAYS = 15         
PATIENCE = 40

POP_SIZE = 10000      
GENERATIONS = 200     
MUTATION_RATE = 0.12
ELITE_SIZE = 250       
MDD_LIMIT = 0.40      

# [2. 유전자 범위] — 탐색 공간 설계 의도
# ┌─ 설계 원칙 ──────────────────────────────────────────────────────────────┐
# │ r_sl_mult  : SL = r_sl_mult / atr_pct                                   │
# │   → ATR 역비례 적응형 손절. 변동성 높으면 SL 좁고, 낮으면 SL 넓어짐    │
# │ t_sl_base  : 추세 모드도 동일한 역비례 구조 (normal/strong 별도 유지)    │
# │ RSI 분리   : normal/strong 별도 기준 → 추세 강도마다 다른 진입 필터     │
# │ atr_inter  : ATR 계산 최적 타임프레임도 GA가 탐색                       │
# └──────────────────────────────────────────────────────────────────────────┘
GENE_BOUNDS = {
    # ── 횡보(Range) ──────────────────────────────────────────────────────────
    'r_adx_limit'        : (15.0, 35.0),
    'r_slope_max'        : (-5.0,  0.0),
    'r_tp_mult'          : ( 1.5,  4.5),
    'r_sl_mult'          : (0.0001, 0.01),   # SL = r_sl_mult / atr_pct (역비례)
    'r_vol_limit'        : ( 0.1,  2.0),
    'rsi_low'            : (30.0, 55.0),
    'rsi_high'           : (45.0, 70.0),

    # ── 일반추세(Normal) ─────────────────────────────────────────────────────
    't_adx_limit_normal' : (20.0, 50.0),
    't_slope_min'        : ( 1.0, 15.0),
    't_tp_short_mult'    : ( 1.2,  5.0),
    't_vol_limit_normal' : ( 0.1,  2.0),
    't_sl_base_normal'   : (0.0001, 0.006),  # SL = t_sl_base_normal / atr_pct
    't_rsi_max_normal'   : (60.0, 85.0),     # 일반추세 롱 RSI 상한 (보수적)
    't_rsi_min_normal'   : (15.0, 40.0),     # 일반추세 숏 RSI 하한 (보수적)

    # ── 강한추세(Strong) ─────────────────────────────────────────────────────
    't_adx_limit_strong' : (35.0, 65.0),
    't_slope_strong'     : ( 3.0, 20.0),
    't_tp_mult'          : ( 4.0, 15.0),
    't_vol_limit_strong' : ( 0.5,  3.5),
    't_sl_base_strong'   : (0.0002, 0.012),  # SL = t_sl_base_strong / atr_pct
    't_rsi_max_strong'   : (70.0, 95.0),     # 강한추세 롱 RSI 상한 (더 허용적)
    't_rsi_min_strong'   : (10.0, 35.0),     # 강한추세 숏 RSI 하한 (더 허용적)

    # ── 트레일링 스탑 ────────────────────────────────────────────────────────
    't_ts_mult'          : (0.0001, 0.005),
    't_sl_activate'      : ( 0.01,  0.05),
}

INTERVALS = ['1h', '2h', '4h']
TF_KEYS   = ['r_inter', 't_inter_normal', 't_inter_strong', 'atr_inter']

def get_data(symbol, interval, days):
    for i in range(3):
        try:
            klines = client.futures_historical_klines(symbol, interval, f"{days} days ago UTC")
            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qv', 'tr', 'tb', 'tq', 'ig'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
            return df[['ts', 'open', 'high', 'low', 'close', 'vol']]
        except: time.sleep(2)
    return pd.DataFrame()

def prepare_full_data():
    print(f"🔄 {SYMBOL} 데이터 로딩 및 지표 계산...")
    df_raw = get_data(SYMBOL, '3m', TOTAL_DAYS + BUFFER_DAYS)
    if df_raw.empty: return None
    df_raw['vol_mean'] = df_raw['vol'].rolling(20).mean()
    for tf in INTERVALS:
        multiplier = 20 if tf == '1h' else 40 if tf == '2h' else 80
        df_raw[f'ma20_{tf}'] = ta.sma(df_raw['close'], length=20 * multiplier)
        df_tf = get_data(SYMBOL, tf, TOTAL_DAYS + BUFFER_DAYS)
        if not df_tf.empty:
            adx_series = ta.adx(df_tf['high'], df_tf['low'], df_tf['close'])['ADX_14']
            df_tf[f'adx_{tf}'] = adx_series
            df_tf[f'adx_slope_{tf}'] = adx_series.pct_change() * 100
            df_tf[f'atr_{tf}'] = ta.atr(df_tf['high'], df_tf['low'], df_tf['close'], length=14)
            df_tf[f'rsi_{tf}'] = ta.rsi(df_tf['close'], length=14)
            df_tf[f'vol_{tf}_mean'] = df_tf['vol'].rolling(20).mean()
            bb = ta.bbands(df_tf['close'], length=20, std=2.0)
            df_tf[f'bbw_{tf}'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / (bb.iloc[:, 1] + 1e-9)
            df_tf[f'bbw_slope_{tf}'] = df_tf[f'bbw_{tf}'].pct_change() * 100
            df_raw = pd.merge_asof(df_raw.sort_values('ts'), df_tf[['ts', f'adx_{tf}', f'adx_slope_{tf}', f'atr_{tf}', f'rsi_{tf}', f'vol_{tf}_mean', f'bbw_{tf}', f'bbw_slope_{tf}']].sort_values('ts'), on='ts', direction='backward')
            df_raw[f'cum_vol_{tf}'] = df_raw.groupby(df_raw['ts'].dt.floor(tf.lower().replace('m', 'min')))['vol'].transform('cumsum')
    return df_raw.dropna().reset_index(drop=True)

def evaluate(args):
    ind_vals, df_main = args
    ind = ind_vals if isinstance(ind_vals, dict) else dict(zip(GENE_BOUNDS.keys(), ind_vals))
    r_tf  = ind.get('r_inter',        '1h')
    tn_tf = ind.get('t_inter_normal', '2h')
    ts_tf = ind.get('t_inter_strong', '1h')
    atr_tf = ind.get('atr_inter',     '4h')   # GA가 최적 ATR 타임프레임 탐색
    bal, peak, mdd, pos, pos_duration = 100.0, 100.0, 0.0, None, 0
    stats = {'range': {'wins': 0, 'trades': 0}, 'trend_normal': {'wins': 0, 'trades': 0}, 'trend_strong': {'wins': 0, 'trades': 0}, 'gross_p': 0.0, 'gross_l': 1e-9}
    trade_returns = []

    for row in df_main.itertuples():
        curr_p = row.close
        if pos is None:
            side, mode = None, None
            # ── 횡보 진입 ───────────────────────────────────────────────────
            if getattr(row, f"adx_{r_tf}") < ind['r_adx_limit'] and \
               getattr(row, f"adx_slope_{r_tf}") <= ind['r_slope_max'] and \
               getattr(row, f"bbw_slope_{r_tf}") < 0:
                if row.vol > (row.vol_mean * ind['r_vol_limit']):
                    rsi_v, ma_v = getattr(row, f"rsi_{r_tf}"), getattr(row, f"ma20_{r_tf}")
                    side = 'long'  if (rsi_v < ind['rsi_low']  and curr_p < ma_v) else \
                           'short' if (rsi_v > ind['rsi_high'] and curr_p > ma_v) else None
                    if side: mode = 'range'
            # ── 강한추세 진입 ───────────────────────────────────────────────
            if not mode:
                if getattr(row, f"adx_{ts_tf}") > ind['t_adx_limit_strong'] and \
                   getattr(row, f"adx_slope_{ts_tf}") >= ind['t_slope_strong']:
                    if getattr(row, f"cum_vol_{ts_tf}") > (getattr(row, f"vol_{ts_tf}_mean") * ind['t_vol_limit_strong']):
                        rsi_v, ma_v = getattr(row, f"rsi_{ts_tf}"), getattr(row, f"ma20_{ts_tf}")
                        # 강한추세: RSI 기준 더 허용적 (극단까지 진입 가능)
                        side = 'long'  if (curr_p > ma_v and rsi_v < ind['t_rsi_max_strong']) else \
                               'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_strong']) else None
                        if side: mode = 'trend_strong'
                if not mode:
                    if getattr(row, f"adx_{tn_tf}") > ind['t_adx_limit_normal'] and \
                       getattr(row, f"adx_slope_{tn_tf}") >= ind['t_slope_min']:
                        if getattr(row, f"cum_vol_{tn_tf}") > (getattr(row, f"vol_{tn_tf}_mean") * ind['t_vol_limit_normal']):
                            rsi_v, ma_v = getattr(row, f"rsi_{tn_tf}"), getattr(row, f"ma20_{tn_tf}")
                            # 일반추세: RSI 기준 보수적 (과열 전 진입)
                            side = 'long'  if (curr_p > ma_v and rsi_v < ind['t_rsi_max_normal']) else \
                                   'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_normal']) else None
                            if side: mode = 'trend_normal'
            if mode and side:
                atr_pct = getattr(row, f"atr_{atr_tf}") / (curr_p + 1e-9)
                pos_duration = 0
                if mode == 'range':
                    tp_pct = atr_pct * ind['r_tp_mult']
                    # ATR 역비례 적응형 손절: 변동성 높으면 SL 좁고, 낮으면 넓음
                    sl_pct = min(ind['r_sl_mult'] / (atr_pct + 1e-9), 0.02)
                elif mode == 'trend_strong':
                    tp_pct = atr_pct * ind['t_tp_mult']
                    sl_pct = min(ind['t_sl_base_strong'] / (atr_pct + 1e-9), 0.05)
                else:  # trend_normal
                    tp_pct = atr_pct * ind['t_tp_short_mult']
                    sl_pct = min(ind['t_sl_base_normal'] / (atr_pct + 1e-9), 0.05)

                if side == 'long':
                    tp, sl = curr_p * (1 + tp_pct), curr_p * (1 - sl_pct)
                else:
                    tp, sl = curr_p * (1 - tp_pct), curr_p * (1 + sl_pct)
                pos = {'side': side, 'ent_p': curr_p, 'sl': sl, 'tp': tp, 'mode': mode}
        else:
            pos_duration += 1
            is_exit, exit_p = False, curr_p

            # ─── 트레일링 스탑 업데이트 (추세 모드 전용) ───────────────────────
            # range 모드는 타임아웃으로 관리하므로 트레일링 불필요
            if pos['mode'] != 'range':
                entry_p = pos['ent_p']
                if pos['side'] == 'long':
                    # 현재 수익률이 발동 기준(t_sl_activate)을 초과했을 때만 트레일링 시작
                    current_pnl_pct = (curr_p - entry_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        # 새로운 트레일링 SL = 현재가에서 t_ts_mult만큼 아래
                        trailing_sl = curr_p * (1 - ind['t_ts_mult'])
                        # SL은 위쪽으로만 이동 (절대 내려가면 안 됨)
                        if trailing_sl > pos['sl']:
                            pos['sl'] = trailing_sl
                else:  # short
                    current_pnl_pct = (entry_p - curr_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        # 새로운 트레일링 SL = 현재가에서 t_ts_mult만큼 위
                        trailing_sl = curr_p * (1 + ind['t_ts_mult'])
                        # SL은 아래쪽으로만 이동 (절대 올라가면 안 됨)
                        if trailing_sl < pos['sl']:
                            pos['sl'] = trailing_sl
            # ────────────────────────────────────────────────────────────────────

            if pos['mode'] == 'range' and pos_duration >= 15: is_exit = True
            elif (curr_p <= pos['sl'] if pos['side'] == 'long' else curr_p >= pos['sl']): is_exit, exit_p = True, pos['sl']
            elif (curr_p >= pos['tp'] if pos['side'] == 'long' else curr_p <= pos['tp']): is_exit, exit_p = True, pos['tp']
            if is_exit:
                # 잔고가 BAL_CAP 초과 시 실제 잔고의 10%만 포지션에 사용
                # → 복리 폭발 방지 + 현실적인 리스크 관리 시뮬레이션
                effective_bal = bal * 0.1 if bal > BAL_CAP else bal
                pnl = effective_bal * (((exit_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - exit_p)/pos['ent_p']) - (TAKER_FEE*2 + SLIPPAGE)) * LEVERAGE
                bal += pnl
                if bal > peak: peak = bal
                mdd = max(mdd, (peak - bal) / (peak + 1e-9))
                stats[pos['mode']]['trades'] += 1
                if pnl > 0: stats[pos['mode']]['wins'] += 1; stats['gross_p'] += pnl
                else: stats['gross_l'] += abs(pnl)
                trade_returns.append(pnl)   # 거래별 손익 기록
                pos = None
                if mdd > MDD_LIMIT or bal <= 5.0: break

    r_tr, n_tr, s_tr = stats['range']['trades'], stats['trend_normal']['trades'], stats['trend_strong']['trades']
    total_trades = r_tr + n_tr + s_tr

    # ─── [개선된 피트니스 함수] ────────────────────────────────────────────────
    # 기본 탈락 조건: 모드별 최소 5거래 미달 또는 MDD 초과
    # (기존 전체 5회 → 모드별 5회로 강화해서 3가지 전략이 모두 작동함을 보장)
    if r_tr < 5 or n_tr < 5 or s_tr < 5 or mdd > MDD_LIMIT:
        return {**ind,
                'Fitness': -1000000.0, 'ROI': bal - 100, 'PF': 0.0, 'MDD': mdd, 'Trades': total_trades}

    pf = (stats['gross_p'] / stats['gross_l']) if stats['gross_l'] > 0 else 1.0
    roi = bal - 100

    # 1) Calmar Ratio: 수익률 ÷ MDD
    #    MDD가 클수록 불리하게 설계 → 수익 대비 리스크를 측정
    #    단순 ROI와 달리 큰 손실 구간을 가진 파라미터를 강력히 패널티
    calmar = roi / (mdd * 100 + 1e-9)

    # 2) 거래 일관성 (샤프 비율 개념)
    #    평균 수익이 높고 편차가 작을수록 높은 점수
    #    "99번 손실 + 1번 대박" 유형의 파라미터는 std가 매우 커서 점수 하락
    mean_ret = np.mean(trade_returns)
    std_ret  = np.std(trade_returns) + 1e-9
    consistency = mean_ret / std_ret

    # 3) 거래 횟수 가중치 (로그 스케일)
    #    거래가 많을수록 통계적 신뢰도가 높다고 판단
    #    단, 선형이 아닌 로그로 취해서 "거래 수 늘리기" 식의 어뷰징 방지
    trade_weight = np.log(total_trades + 1)

    # 최종 피트니스: 세 요소의 곱
    #  - calmar > 0 (흑자) 이고 consistency > 0 (평균 거래가 수익) 일 때만 양수
    #  - 둘 중 하나라도 음수면 전체 음수 → 자동 탈락
    #  - trade_weight는 항상 양수이므로 방향은 바꾸지 않고 크기만 조정
    if calmar > 0:
        fitness = calmar * consistency * trade_weight
    else:
        # 손실 구간: calmar만 사용해서 최소한의 순위 구분
        fitness = calmar * trade_weight

    return {**ind,   # 파라미터 먼저 → 아래 계산값이 덮어씀 (순서 중요)
            'Fitness': fitness, 'ROI': roi, 'PF': pf, 'MDD': mdd, 'Trades': total_trades,
            'Calmar': round(calmar, 4), 'Consistency': round(consistency, 4)}
    # ──────────────────────────────────────────────────────────────────────────

# [4. GA 루프 (PATIENCE 추가)]
def run_ga(df_train):
    population = [{**{k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()},
                   **{tf: random.choice(INTERVALS) for tf in TF_KEYS}}
                  for _ in range(POP_SIZE)]
    best_overall_fitness, no_improvement_count, best_individual = -float('inf'), 0, None
    
    for gen in range(GENERATIONS):
        with Pool(cpu_count()) as p:
            results = p.map(evaluate, [(ind, df_train) for ind in population])
        results.sort(key=lambda x: x['Fitness'], reverse=True)
        
        if results[0]['Fitness'] > best_overall_fitness:
            best_overall_fitness, best_individual, no_improvement_count = results[0]['Fitness'], results[0], 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= PATIENCE:
            print(f"   ⏱️ {PATIENCE}세대 정체로 조기 종료 (세대: {gen+1})")
            break
            
        elites = results[:ELITE_SIZE]
        new_pop = [{k: v for k, v in e.items() if k in list(GENE_BOUNDS.keys()) + TF_KEYS} for e in elites]
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = {k: (random.choice([p1[k], p2[k]]) if random.random() > MUTATION_RATE else random.uniform(GENE_BOUNDS[k][0], GENE_BOUNDS[k][1])) for k in GENE_BOUNDS.keys()}
            for tf in TF_KEYS: child[tf] = random.choice([p1[tf], p2[tf]])
            new_pop.append(child)
        population = new_pop
    return best_individual

if __name__ == "__main__":
    import math, os

    # ── 저장 파일 경로 ────────────────────────────────────────────────────────
    FILE_PARAMS = "WFA_Params.csv"       # 채택된 파라미터 (윈도우마다 append)
    FILE_OOS    = "WFA_OOS_Summary.csv"  # 전체 윈도우 요약 (윈도우마다 append)

    df_all = prepare_full_data()
    if df_all is not None:

        # ── 이어쓰기 지원: 기존 파일에서 마지막 윈도우 번호와 OOS 상태 복원 ──
        if os.path.exists(FILE_OOS):
            prev = pd.read_csv(FILE_OOS)
            win_idx  = int(prev['window'].max())
            oos_bal  = float(prev['oos_bal'].iloc[-1])
            oos_peak = float(prev['oos_bal'].max())
            oos_mdd  = float(prev['oos_mdd_cumul'].iloc[-1]) if 'oos_mdd_cumul' in prev.columns else 0.0

            # 마지막으로 완료된 윈도우의 test_end 다음 날부터 재시작
            last_test_end = pd.to_datetime(prev['test_end'].iloc[-1])
            # test_end = train_end + TEST_DAYS → train_start 역산
            current_train_start = last_test_end - pd.Timedelta(days=TEST_DAYS_PER_WIN + TRAIN_DAYS - STEP_DAYS)
            print(f"⏩ 기존 파일 발견 → 윈도우 {win_idx}부터 이어서 시작")
        else:
            win_idx  = 0
            oos_bal  = 100.0
            oos_peak = 100.0
            oos_mdd  = 0.0
            current_train_start = df_all['ts'].min()
            print(f"🆕 새로 시작")

        OVERFIT_THRESHOLD = 0.3

        while True:
            train_end = current_train_start + pd.Timedelta(days=TRAIN_DAYS)
            test_end  = train_end + pd.Timedelta(days=TEST_DAYS_PER_WIN)
            if test_end > df_all['ts'].max(): break

            win_idx += 1
            print(f"\n{'='*60}")
            print(f"📂 윈도우 {win_idx} | 학습: {current_train_start.date()} ~ {train_end.date()} "
                  f"| 테스트: {train_end.date()} ~ {test_end.date()}")

            # ── STEP 1: 학습 구간 GA 최적화 ──────────────────────────────────
            df_train = df_all[(df_all['ts'] >= current_train_start) & (df_all['ts'] < train_end)]
            best_params = run_ga(df_train)

            if not best_params or best_params['Fitness'] <= -1000:
                print(f"   ⚠️ 유효한 파라미터를 찾지 못함 → 이 윈도우 건너뜀")
                current_train_start += pd.Timedelta(days=STEP_DAYS)
                continue

            train_roi = best_params['ROI']
            print(f"   [학습] ROI: {min(train_roi, BAL_CAP-100):.2f}% | MDD: {best_params['MDD']*100:.1f}% | "
                  f"Calmar: {best_params.get('Calmar', 0):.3f} | "
                  f"Consistency: {best_params.get('Consistency', 0):.3f} | "
                  f"Trades: {best_params['Trades']}")

            # ── STEP 2: 테스트 구간 실전 평가 ────────────────────────────────
            df_test = df_all[(df_all['ts'] >= train_end) & (df_all['ts'] < test_end)]
            test_result = evaluate((best_params, df_test))
            test_roi    = test_result['ROI']
            test_mdd    = test_result['MDD']
            test_trades = test_result['Trades']

            # ── STEP 3: 과적합 비율 판정 (로그 스케일) ───────────────────────
            if train_roi > 0:
                log_train     = math.log1p(max(train_roi, 0) / 100)
                log_test      = math.log1p(test_roi / 100) if test_roi > -100 else -1.0
                overfit_ratio = log_test / (log_train + 1e-9)
            else:
                overfit_ratio = -1.0

            is_accepted = overfit_ratio >= OVERFIT_THRESHOLD
            status = "✅ 채택" if is_accepted else "❌ 기각"
            print(f"   [테스트] ROI: {test_roi:.2f}% | MDD: {test_mdd*100:.1f}% | "
                  f"Trades: {test_trades} | 과적합비율: {overfit_ratio:.3f} → {status}")

            # ── STEP 4: OOS 누적 곡선 업데이트 ──────────────────────────────
            if test_trades > 0:
                oos_bal *= max(1 + test_roi / 100, 0)  # 음수 방지
                if oos_bal > oos_peak: oos_peak = oos_bal
                oos_mdd = max(oos_mdd, (oos_peak - oos_bal) / (oos_peak + 1e-9))

            # ── STEP 5: 윈도우 완료 → 즉시 파일에 기록 ──────────────────────
            # OOS 요약 한 줄 append
            oos_row = pd.DataFrame([{
                'window'        : win_idx,
                'train_start'   : current_train_start.date(),
                'train_end'     : train_end.date(),
                'test_end'      : test_end.date(),
                'train_roi'     : round(train_roi, 4),
                'test_roi'      : round(test_roi, 4),
                'test_mdd'      : round(test_mdd, 4),
                'test_trades'   : test_trades,
                'overfit_ratio' : round(overfit_ratio, 4),
                'is_accepted'   : is_accepted,
                'oos_bal'       : round(oos_bal, 4),
                'oos_mdd_cumul' : round(oos_mdd, 4),
            }])
            # header=True는 파일이 없을 때만, 이후엔 header=False로 append
            oos_row.to_csv(FILE_OOS, mode='a',
                           header=not os.path.exists(FILE_OOS) or win_idx == 1,
                           index=False, encoding='utf-8-sig')

            # 채택된 파라미터 append
            if is_accepted:
                param_row = best_params.copy()
                param_row.update({
                    'win_idx'       : win_idx,
                    'train_roi'     : round(train_roi, 4),
                    'test_roi'      : round(test_roi, 4),
                    'overfit_ratio' : round(overfit_ratio, 4),
                    'start_date'    : current_train_start.date(),
                    'end_date'      : train_end.date(),
                })
                param_df = pd.DataFrame([param_row])
                param_df.to_csv(FILE_PARAMS, mode='a',
                                header=not os.path.exists(FILE_PARAMS),
                                index=False, encoding='utf-8-sig')

            print(f"   💾 저장 완료 → {FILE_OOS}"
                  + (f" / {FILE_PARAMS}" if is_accepted else ""))

            current_train_start += pd.Timedelta(days=STEP_DAYS)

        # ── 최종 리포트 ───────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"🏁 WFA 완료")
        if os.path.exists(FILE_OOS):
            df_oos = pd.read_csv(FILE_OOS)
            total_wins = int(df_oos['window'].max())
            accepted   = df_oos['is_accepted'].sum()
            print(f"   총 {total_wins}개 윈도우 | 채택: {accepted}개 ({accepted/total_wins*100:.1f}%)")
            print(f"\n📈 [OOS 누적 성과]")
            print(f"   최종 OOS 잔고: ${df_oos['oos_bal'].iloc[-1]:,.2f}  (시작: $100.00)")
            print(f"   OOS 총 수익률: {df_oos['oos_bal'].iloc[-1] - 100:+.2f}%")
            print(f"   OOS 최대 낙폭: {df_oos['oos_mdd_cumul'].max()*100:.2f}%")
            print(f"\n📋 [윈도우별 요약]")
            print(f"{'Win':>4} | {'Train ROI':>9} | {'Test ROI':>8} | {'OvFit':>6} | {'채택':>4} | OOS잔고")
            print("─" * 60)
            for _, r in df_oos.iterrows():
                flag = "✅" if r['is_accepted'] else "❌"
                print(f"  {int(r['window']):2d}  | {r['train_roi']:+8.2f}%  | {r['test_roi']:+7.2f}%  | "
                      f"{r['overfit_ratio']:6.3f} | {flag}   | ${r['oos_bal']:,.2f}")
        print(f"\n💾 전체 결과: {FILE_OOS}")
        if os.path.exists(FILE_PARAMS):
            print(f"💾 채택 파라미터: {FILE_PARAMS}")
        print("="*60)
