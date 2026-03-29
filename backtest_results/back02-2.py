import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import warnings
import random
from multiprocessing import Pool, cpu_count
import os
import time
import pprint

warnings.filterwarnings('ignore')

# [1. 설정 영역]
client = Client("", "", {"verify": True, "timeout": 20}) 
SYMBOL = 'FARTCOINUSDT'
TOTAL_DAYS = 480      
BUFFER_DAYS = 60      
LEVERAGE = 5          

TAKER_FEE = 0.0005
SLIPPAGE = 0.001

# 잔고 상한선: 복리 폭발 방지
BAL_CAP = 10_000.0

# GA 하이퍼 파라미터
POP_SIZE = 10000        
GENERATIONS = 500     
MUTATION_RATE = 0.12  
ELITE_SIZE = 250      
PATIENCE = 40         
MDD_LIMIT = 0.6       

# 유전자 범위 — 설계 의도
# ┌─ 설계 원칙 ──────────────────────────────────────────────────────────────┐
# │ r_sl_mult / t_sl_base : SL = base / atr_pct (ATR 역비례 적응형 손절)    │
# │   → 변동성 높을수록 SL 좁아짐, 낮을수록 넓어짐                          │
# │ RSI 분리 : normal/strong 별도 기준 → 추세 강도별 다른 진입 필터         │
# │ atr_inter : ATR 계산 최적 타임프레임도 GA가 탐색                        │
# └──────────────────────────────────────────────────────────────────────────┘
GENE_BOUNDS = {
    # ── 횡보(Range) ──────────────────────────────────────────────────────────
    'r_adx_limit'        : (15.0, 35.0),
    'r_slope_max'        : (-5.0,  0.0),
    'r_tp_mult'          : ( 1.5,  4.5),
    'r_sl_mult'          : (0.0001, 0.01),
    'r_vol_limit'        : ( 0.1,  2.0),
    'rsi_low'            : (30.0, 55.0),
    'rsi_high'           : (45.0, 70.0),

    # ── 일반추세(Normal) ─────────────────────────────────────────────────────
    't_adx_limit_normal' : (20.0, 50.0),
    't_slope_min'        : ( 1.0, 15.0),
    't_tp_short_mult'    : ( 1.2,  5.0),
    't_vol_limit_normal' : ( 0.1,  2.0),
    't_sl_base_normal'   : (0.0001, 0.006),
    't_rsi_max_normal'   : (60.0, 85.0),
    't_rsi_min_normal'   : (15.0, 40.0),

    # ── 강한추세(Strong) ─────────────────────────────────────────────────────
    't_adx_limit_strong' : (35.0, 65.0),
    't_slope_strong'     : ( 3.0, 20.0),
    't_tp_mult'          : ( 4.0, 15.0),
    't_vol_limit_strong' : ( 0.5,  3.5),
    't_sl_base_strong'   : (0.0002, 0.012),
    't_rsi_max_strong'   : (70.0, 95.0),
    't_rsi_min_strong'   : (10.0, 35.0),

    # ── 트레일링 스탑 ────────────────────────────────────────────────────────
    't_ts_mult'          : (0.0001, 0.005),
    't_sl_activate'      : ( 0.01,  0.05),
}

INTERVALS = ['1h', '2h', '4h']
TF_KEYS   = ['r_inter', 't_inter_normal', 't_inter_strong', 'atr_inter']

# [2. 데이터 준비]
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
            bbl, bbm, bbu = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
            df_tf[f'bbw_{tf}'] = (bbu - bbl) / (bbm + 1e-9)
            df_tf[f'bbw_slope_{tf}'] = df_tf[f'bbw_{tf}'].pct_change() * 100
            
            df_raw = pd.merge_asof(df_raw.sort_values('ts'), 
                                   df_tf[['ts', f'adx_{tf}', f'adx_slope_{tf}', f'atr_{tf}', f'rsi_{tf}', 
                                          f'vol_{tf}_mean', f'bbw_{tf}', f'bbw_slope_{tf}']].sort_values('ts'), 
                                   on='ts', direction='backward')
            df_raw[f'cum_vol_{tf}'] = df_raw.groupby(df_raw['ts'].dt.floor(tf.lower().replace('m', 'min')))['vol'].transform('cumsum')
    return df_raw.dropna().reset_index(drop=True)

# [3. 백테스트 엔진]
def evaluate(args):
    ind_vals, df_main = args
    ind = ind_vals if isinstance(ind_vals, dict) else dict(zip(GENE_BOUNDS.keys(), ind_vals))
    r_tf  = ind.get('r_inter',        '1h')
    tn_tf = ind.get('t_inter_normal', '2h')
    ts_tf = ind.get('t_inter_strong', '1h')
    atr_tf = ind.get('atr_inter',     '4h')   # GA가 최적 ATR 타임프레임 탐색

    bal, peak, mdd = 100.0, 100.0, 0.0
    pos = None
    pos_duration = 0
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
                        # 강한추세: RSI 기준 더 허용적
                        side = 'long'  if (curr_p > ma_v and rsi_v < ind['t_rsi_max_strong']) else \
                               'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_strong']) else None
                        if side: mode = 'trend_strong'
                # ── 일반추세 진입 ───────────────────────────────────────────
                if not mode:
                    if getattr(row, f"adx_{tn_tf}") > ind['t_adx_limit_normal'] and \
                       getattr(row, f"adx_slope_{tn_tf}") >= ind['t_slope_min']:
                        if getattr(row, f"cum_vol_{tn_tf}") > (getattr(row, f"vol_{tn_tf}_mean") * ind['t_vol_limit_normal']):
                            rsi_v, ma_v = getattr(row, f"rsi_{tn_tf}"), getattr(row, f"ma20_{tn_tf}")
                            # 일반추세: RSI 기준 보수적
                            side = 'long'  if (curr_p > ma_v and rsi_v < ind['t_rsi_max_normal']) else \
                                   'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_normal']) else None
                            if side: mode = 'trend_normal'

            if mode and side:
                atr_pct = getattr(row, f"atr_{atr_tf}") / (curr_p + 1e-9)
                pos_duration = 0

                if mode == 'range':
                    tp_pct = atr_pct * ind['r_tp_mult']
                    # ATR 역비례 적응형 손절
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
            is_exit = False
            exit_p = curr_p

            # ─── 트레일링 스탑 업데이트 (추세 모드 전용) ───────────────────────
            # range 모드는 타임아웃(15봉)으로 관리하므로 트레일링 적용 제외
            if pos['mode'] != 'range':
                entry_p = pos['ent_p']
                if pos['side'] == 'long':
                    # 현재 수익률이 t_sl_activate 기준을 초과했을 때만 트레일링 발동
                    current_pnl_pct = (curr_p - entry_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        # 현재가 아래 t_ts_mult 지점을 새 손절선으로 설정
                        trailing_sl = curr_p * (1 - ind['t_ts_mult'])
                        # SL은 반드시 위 방향으로만 이동 (한 번 올린 SL은 절대 내리지 않음)
                        if trailing_sl > pos['sl']:
                            pos['sl'] = trailing_sl
                else:  # short
                    current_pnl_pct = (entry_p - curr_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        # 현재가 위 t_ts_mult 지점을 새 손절선으로 설정
                        trailing_sl = curr_p * (1 + ind['t_ts_mult'])
                        # SL은 반드시 아래 방향으로만 이동 (한 번 내린 SL은 절대 올리지 않음)
                        if trailing_sl < pos['sl']:
                            pos['sl'] = trailing_sl
            # ────────────────────────────────────────────────────────────────────

            # 횡보 타임아웃 체크
            if pos['mode'] == 'range' and pos_duration >= 15:
                is_exit = True
                exit_p = curr_p
            
            # 익손절 체크
            if not is_exit:
                if (curr_p <= pos['sl'] if pos['side'] == 'long' else curr_p >= pos['sl']):
                    is_exit = True
                    exit_p = pos['sl']
                elif (curr_p >= pos['tp'] if pos['side'] == 'long' else curr_p <= pos['tp']):
                    is_exit = True
                    exit_p = pos['tp']

            if is_exit:
                # 잔고가 BAL_CAP 초과 시 실제 잔고의 10%만 포지션에 사용
                effective_bal = bal * 0.1 if bal > BAL_CAP else bal
                pnl = effective_bal * (((exit_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - exit_p)/pos['ent_p']) - (TAKER_FEE*2 + SLIPPAGE)) * LEVERAGE
                bal += pnl
                bal = min(bal, BAL_CAP)   # 복리 폭발 방지
                if bal > peak: peak = bal
                mdd = max(mdd, (peak - bal) / (peak + 1e-9))
                stats[pos['mode']]['trades'] += 1
                if pnl > 0: stats[pos['mode']]['wins'] += 1; stats['gross_p'] += pnl
                else: stats['gross_l'] += abs(pnl)
                trade_returns.append(pnl)   # 거래별 손익 기록
                pos = None
                if mdd > MDD_LIMIT or bal <= 5.0: break

    # ─── 평가 지표 산출 ────────────────────────────────────────────────────────
    r_tr, n_tr, s_tr = stats['range']['trades'], stats['trend_normal']['trades'], stats['trend_strong']['trades']
    total_trades = r_tr + n_tr + s_tr

    # [개선] 탈락 조건: 모드별 최소 5거래 미달 또는 MDD 초과
    # 기존의 "승률 100%" 필터를 제거하고 최소 거래 수 기준을 3 → 5로 상향
    # 이유: wr==1.0 조건은 좋은 파라미터도 운 좋게 3번만 이긴 경우 걸러버림
    #       최소 거래 수를 높여서 통계적 신뢰도로 대체하는 것이 더 합리적
    if r_tr < 5 or n_tr < 5 or s_tr < 5 or mdd > MDD_LIMIT:
        return {**ind,
                'Fitness': -1000000.0, 'ROI': bal - 100, 'PF': 0.0, 'MDD': mdd,
                'Trades': total_trades, 'R_Tr': r_tr, 'TN_Tr': n_tr, 'TS_Tr': s_tr}

    pf = (stats['gross_p'] / stats['gross_l']) if stats['gross_l'] > 0 else 1.0
    roi = bal - 100

    # ─── [개선된 피트니스 함수] ────────────────────────────────────────────────
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
        fitness = calmar * trade_weight

    return {**ind,
            'Fitness': fitness, 'ROI': roi, 'PF': pf, 'MDD': mdd, 'Trades': total_trades,
            'Calmar': round(calmar, 4), 'Consistency': round(consistency, 4),
            'R_Tr': r_tr, 'TN_Tr': n_tr, 'TS_Tr': s_tr}
    # ──────────────────────────────────────────────────────────────────────────

# [4. GA 메인 루프]
def run_ga(df_train):
    population = [{**{k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()},
                   **{tf: random.choice(INTERVALS) for tf in TF_KEYS}}
                  for _ in range(POP_SIZE)]
    best_overall_fitness, no_improvement_count, best_overall_result = -float('inf'), 0, None
    print(f"🧬 GA 최적화 시작: POP={POP_SIZE}, SYMBOL={SYMBOL}")

    for gen in range(GENERATIONS):
        with Pool(cpu_count()) as p:
            results = p.map(evaluate, [(ind, df_train) for ind in population])
        results.sort(key=lambda x: x['Fitness'], reverse=True)
        current_best = results[0]

        if current_best['Fitness'] > best_overall_fitness:
            best_overall_fitness, best_overall_result, no_improvement_count = current_best['Fitness'], current_best, 0
        else: no_improvement_count += 1

        print(f" 세대 {gen+1:3d} | Fitness: {current_best['Fitness']:.4f} | ROI: {current_best['ROI']:.2f}% | "
              f"MDD: {current_best['MDD']*100:.1f}% | Calmar: {current_best.get('Calmar', 0):.3f} | "
              f"Consistency: {current_best.get('Consistency', 0):.3f} | 정체: {no_improvement_count}/{PATIENCE}")
        if no_improvement_count >= PATIENCE: break

        elites = results[:ELITE_SIZE]
        new_pop = [{k: v for k, v in e.items() if k in list(GENE_BOUNDS.keys()) + TF_KEYS} for e in elites]
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = {k: (random.uniform(GENE_BOUNDS[k][0], GENE_BOUNDS[k][1]) if random.random() < MUTATION_RATE else random.choice([p1[k], p2[k]])) for k in GENE_BOUNDS.keys()}
            for tf in TF_KEYS: child[tf] = random.choice([p1[tf], p2[tf]])
            new_pop.append(child)
        population = new_pop
    return best_overall_result

if __name__ == "__main__":
    df_all = prepare_full_data()
    if df_all is not None:
        print(f"\n🚀 {SYMBOL} 딥-최적화 시작 (트레일링 스탑 적용)")
        best_params = run_ga(df_all)
        print("\n" + "="*50)
        print("🏆 최적 파라미터 결과")
        pprint.pprint(best_params)
        pd.DataFrame([best_params]).to_csv("Final_Optimized_Params.csv", index=False, encoding='utf-8-sig')
        print("="*50)
