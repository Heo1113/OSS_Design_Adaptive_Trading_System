import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import warnings
import os
import time

warnings.filterwarnings('ignore')

# [1. 설정 영역]
client = Client("", "") 
SYMBOL = 'FARTCOINUSDT'
TEST_DAYS = 480        
BUFFER_DAYS = 60
LEVERAGE = 5

TAKER_FEE = 0.0005   
MAKER_FEE = 0.0002    
SLIPPAGE = 0.001

# 포지션 축소 기준선: 잔고가 이 값을 넘으면 전체 잔고가 아닌 10%만 포지션에 사용
# 잔고 자체는 계속 쌓임 — back02.py / back02-2.py와 동일한 기준 유지
BAL_CAP = 10_000.0

# 최적 파라미터 (GA 결과값)
BEST_PARAMS = {
    'r_adx_limit': 8.01,         # 매우 낮은 ADX (강한 박스권 지향)
    'r_slope_max': -1.35,
    'r_tp_mult': 3.73,
    'r_sl_mult': 0.00012,
    'r_vol_limit': 0.95,
    'rsi_low': 53.35,
    'rsi_high': 45.24,

    # -- 일반 추세(Normal Trend) 모드 파라미터 --
    't_adx_limit_normal': 20.00,
    't_slope_min': 1.22,
    't_tp_short_mult': 1.35,
    't_vol_limit_normal': 0.10,
    't_sl_base_normal': 0.00012,
    't_rsi_max_normal': 81.19,
    't_rsi_min_normal': 20.01,

    # -- 강한 추세(Strong Trend) 모드 파라미터 --
    't_adx_limit_strong': 35.88,
    't_slope_strong': 3.36,
    't_tp_mult': 12.76,          # 강한 추세에서 이익을 길게 가져감
    't_vol_limit_strong': 0.80,
    't_sl_base_strong': 0.00022,
    't_rsi_max_strong': 75.43,
    't_rsi_min_strong': 19.20,

    # -- 트레일링 스탑 및 타임프레임 --
    't_ts_mult': 0.0012,
    't_sl_activate': 0.011,      # 약 1.1% 수익 시 트레일링 가동
    'r_inter': '2h',
    't_inter_normal': '1h',
    't_inter_strong': '4h',
    'atr_inter': '4h'
}

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

def run_backtest(df_main, ind):
    r_tf  = ind['r_inter']
    tn_tf = ind['t_inter_normal']
    ts_tf = ind['t_inter_strong']
    atr_tf = ind['atr_inter']
    bal, peak_bal, max_dd = 100.0, 100.0, 0.0
    pos = None
    stats = {
        'range': {'trades': 0, 'wins': 0, 'profit': 0.0},
        'trend_normal': {'trades': 0, 'wins': 0, 'profit': 0.0},
        'trend_strong': {'trades': 0, 'wins': 0, 'profit': 0.0}
    }
    trade_log = []
    trade_returns = []  # 일관성 지표 계산용: 거래별 손익 기록
    pos_duration = 0

    for row in df_main.itertuples():
        curr_p = row.close
        
        if pos is None:
            mode, side = None, None

            # 1. 횡보 판정
            if getattr(row, f"adx_{r_tf}") < ind['r_adx_limit'] and \
               getattr(row, f"adx_slope_{r_tf}") <= ind['r_slope_max'] and \
               getattr(row, f"bbw_slope_{r_tf}") < 0:
                if row.vol > (row.vol_mean * ind['r_vol_limit']):
                    rsi_v, ma_v = getattr(row, f"rsi_{r_tf}"), getattr(row, f"ma20_{r_tf}")
                    side = 'long'  if (rsi_v < ind['rsi_low']  and curr_p < ma_v) else \
                           'short' if (rsi_v > ind['rsi_high'] and curr_p > ma_v) else None
                    if side: mode = 'range'

            # 2. 추세 판정
            if not mode:
                if getattr(row, f"adx_{ts_tf}") > ind['t_adx_limit_strong'] and \
                   getattr(row, f"adx_slope_{ts_tf}") >= ind['t_slope_strong']:
                    if getattr(row, f"cum_vol_{ts_tf}") > (getattr(row, f"vol_{ts_tf}_mean") * ind['t_vol_limit_strong']):
                        rsi_v, ma_v = getattr(row, f"rsi_{ts_tf}"), getattr(row, f"ma20_{ts_tf}")
                        side = 'long'  if (curr_p > ma_v and rsi_v < ind['t_rsi_max_strong']) else \
                               'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_strong']) else None
                        if side: mode = 'trend_strong'

                if not mode:
                    if getattr(row, f"adx_{tn_tf}") > ind['t_adx_limit_normal'] and \
                       getattr(row, f"adx_slope_{tn_tf}") >= ind['t_slope_min']:
                        if getattr(row, f"cum_vol_{tn_tf}") > (getattr(row, f"vol_{tn_tf}_mean") * ind['t_vol_limit_normal']):
                            rsi_v, ma_v = getattr(row, f"rsi_{tn_tf}"), getattr(row, f"ma20_{tn_tf}")
                            side = 'long'  if (curr_p > ma_v and rsi_v < ind['t_rsi_max_normal']) else \
                                   'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_normal']) else None
                            if side: mode = 'trend_normal'

            if mode and side:
                atr_pct = getattr(row, f"atr_{atr_tf}") / (curr_p + 1e-9)
                pos_duration = 0

                if mode == 'range':
                    tp_pct = atr_pct * ind['r_tp_mult']
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
                    # 수익률이 t_sl_activate 기준을 넘었을 때만 트레일링 발동
                    current_pnl_pct = (curr_p - entry_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        # 현재가 아래 t_ts_mult 지점을 새 손절선으로 설정
                        trailing_sl = curr_p * (1 - ind['t_ts_mult'])
                        # 반드시 기존 SL보다 높을 때만 갱신 (SL은 위로만 이동)
                        if trailing_sl > pos['sl']:
                            pos['sl'] = trailing_sl
                else:  # short
                    current_pnl_pct = (entry_p - curr_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        # 현재가 위 t_ts_mult 지점을 새 손절선으로 설정
                        trailing_sl = curr_p * (1 + ind['t_ts_mult'])
                        # 반드시 기존 SL보다 낮을 때만 갱신 (SL은 아래로만 이동)
                        if trailing_sl < pos['sl']:
                            pos['sl'] = trailing_sl
            # ────────────────────────────────────────────────────────────────────

            # 횡보 타임아웃 (15봉)
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
                exit_fee = TAKER_FEE
                # 잔고가 BAL_CAP 초과 시 실제 잔고의 10%만 포지션에 사용
                effective_bal = bal * 0.1 if bal > BAL_CAP else bal
                pnl = effective_bal * (((exit_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - exit_p)/pos['ent_p']) - (TAKER_FEE + exit_fee + SLIPPAGE*2)) * LEVERAGE
                bal += pnl
                if bal > peak_bal: peak_bal = bal
                max_dd = max(max_dd, (peak_bal - bal) / (peak_bal + 1e-9))
                stats[pos['mode']]['trades'] += 1
                stats[pos['mode']]['profit'] += pnl
                if pnl > 0: stats[pos['mode']]['wins'] += 1
                trade_log.append({'ts': row.ts, 'profit': pnl, 'bal': bal, 'mode': pos['mode']})
                trade_returns.append(pnl)   # 일관성 지표 계산용
                pos = None
                if bal <= 5.0: break

    return bal, trade_log, stats, peak_bal, max_dd, trade_returns

if __name__ == "__main__":
    total_days = TEST_DAYS + BUFFER_DAYS
    df_main = get_data(SYMBOL, '3m', total_days)
    
    if not df_main.empty:
        df_main['vol_mean'] = df_main['vol'].rolling(20).mean()
        needed_tfs = list(set([
            BEST_PARAMS['r_inter'],
            BEST_PARAMS['t_inter_normal'],
            BEST_PARAMS['t_inter_strong'],
            BEST_PARAMS['atr_inter'],
        ]))
        
        for tf in needed_tfs:
            print(f"🔄 {tf} 데이터 결합 중...")
            df_tf = get_data(SYMBOL, tf, total_days)
            if not df_tf.empty:
                df_tf[f'ma20_{tf}'] = ta.sma(df_tf['close'], length=20)
                adx_res = ta.adx(df_tf['high'], df_tf['low'], df_tf['close'])
                df_tf[f'adx_{tf}'] = adx_res['ADX_14']
                df_tf[f'adx_slope_{tf}'] = adx_res['ADX_14'].pct_change() * 100
                df_tf[f'atr_{tf}'] = ta.atr(df_tf['high'], df_tf['low'], df_tf['close'], length=14)
                df_tf[f'rsi_{tf}'] = ta.rsi(df_tf['close'], length=14)
                df_tf[f'vol_{tf}_mean'] = df_tf['vol'].rolling(20).mean()

                # [버그 수정] bbw_slope 컬럼 추가 (back02-2.py와 동일한 진입 조건 유지)
                bb = ta.bbands(df_tf['close'], length=20, std=2.0)
                bbl, bbm, bbu = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
                df_tf[f'bbw_{tf}'] = (bbu - bbl) / (bbm + 1e-9)
                df_tf[f'bbw_slope_{tf}'] = df_tf[f'bbw_{tf}'].pct_change() * 100
                
                df_main = pd.merge_asof(
                    df_main.sort_values('ts'), 
                    df_tf[['ts', f'ma20_{tf}', f'adx_{tf}', f'adx_slope_{tf}', f'atr_{tf}',
                            f'rsi_{tf}', f'vol_{tf}_mean', f'bbw_{tf}', f'bbw_slope_{tf}']].sort_values('ts'), 
                    on='ts', direction='backward'
                )
                floor_freq = tf.lower().replace('m', 'min')
                df_main[f'cum_vol_{tf}'] = df_main.groupby(df_main['ts'].dt.floor(floor_freq))['vol'].transform('cumsum')

        df_ready = df_main[df_main['ts'] >= (df_main['ts'].max() - pd.Timedelta(days=TEST_DAYS))].dropna().reset_index(drop=True)
        final_bal, logs, mode_stats, peak_bal, max_dd, _ = run_backtest(df_ready, BEST_PARAMS)

        all_modes = ['range', 'trend_normal', 'trend_strong']
        total_trades = sum(mode_stats[m]['trades'] for m in all_modes)
        total_wins = sum(mode_stats[m]['wins'] for m in all_modes)
        
        print("\n" + "="*65)
        print(f"📊 {SYMBOL} 통합 성과 보고서 (트레일링 스탑 + bbw_slope 반영)")
        print(f"💰 최종 잔고: ${final_bal:,.2f} | 수익률: {((final_bal-100)/100)*100:,.1f}%")
        print(f"🔝 최고 잔고: ${peak_bal:,.2f} | 📉 최대 낙폭(MDD): {max_dd*100:.2f}%")
        print(f"🤝 전체 거래: {total_trades}회 | 승률: {(total_wins/total_trades*100) if total_trades>0 else 0:.1f}%")

        print("\n" + "─"*20 + " [모드별 상세 성과] " + "─"*21)
        display_map = {'range': '횡보(Range)', 'trend_normal': '일반추세(Normal)', 'trend_strong': '파워추세(Power)'}
        
        abs_total_profit = sum(abs(mode_stats[m]['profit']) for m in all_modes)
        for m in all_modes:
            m_data = mode_stats[m]
            m_winrate = (m_data['wins']/m_data['trades']*100) if m_data['trades'] > 0 else 0
            contribution = (abs(m_data['profit']) / (abs_total_profit + 1e-9) * 100)
            print(f"▶ {display_map[m]:15}: {m_data['trades']:3}회 거래 | 승률 {m_winrate:5.1f}%")
            print(f"                 순수익 ${m_data['profit']:12,.2f} (비중: {contribution:5.1f}%)")

        if logs:
            df_log = pd.DataFrame(logs)
            print("\n" + "─"*20 + " [월별 수익 추이] " + "─"*24)
            monthly_stats = df_log.set_index('ts')['profit'].resample('ME').sum()
            print(monthly_stats)

            print("\n" + "─"*20 + " [치명적 손실 분석] " + "─"*21)
            losses = df_log[df_log['profit'] < 0]['profit']
            if not losses.empty:
                print(f"📉 평균 손실액: ${abs(losses.mean()):,.2f}")
                print(f"😱 최대 단일 손실: ${abs(losses.min()):,.2f}")

            print(f"\n⚠️ [마지막 거래 기록 (청산 원인 파악)]")
            for last_trade in logs[-3:]:
                print(f"   {last_trade['ts']} | 수익: ${last_trade['profit']:,.2f} | 잔고: ${last_trade['bal']:,.2f} ({display_map[last_trade['mode']]})")
        print("="*65)
