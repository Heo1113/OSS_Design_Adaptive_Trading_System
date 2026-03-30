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
TEST_DAYS = 90        
BUFFER_DAYS = 60
LEVERAGE = 5

TAKER_FEE = 0.0005   
MAKER_FEE = 0.0002    
SLIPPAGE = 0.001

BAL_CAP = 10_000.0

# 최적 파라미터 (GA 결과값)
BEST_PARAMS = {
    # -- 타임프레임 (Interval) --
    'r_inter': '4h',             # 횡보 판단은 4시간봉으로 묵직하게 (단, RSI 진입은 3분봉 적용)
    't_inter_normal': '1h',      # 일반 추세는 1시간봉 기준
    't_inter_strong': '2h',      # 강한 추세는 2시간봉 기준
    'atr_inter': '4h',           # 변동성(손절/익절폭)은 4시간 기준

    # -- 횡보(Range) 모드 --
    'r_adx_limit': 31.703924723061125, 
    'r_slope_max': -0.16416230433050583, 
    'r_tp_mult': 3.350192999507061, 
    'r_sl_mult': 0.00010950137271018771,
    'r_vol_limit': 0.9226953135073943, 
    'rsi_low': 30.701162047170136, 
    'rsi_high': 70.00471016074503,

    # -- 일반 추세(Normal Trend) --
    't_adx_limit_normal': 20.466508133399195, 
    't_slope_min': 1.4639557623620882, 
    't_tp_short_mult': 4.126617769206629,
    't_vol_limit_normal': 0.12849400259095464, 
    't_sl_base_normal': 0.00010084196466677202,
    't_rsi_max_normal': 80.12792284983662, 
    't_rsi_min_normal': 16.62446590698339,

    # -- 강한 추세(Strong Trend) --
    't_adx_limit_strong': 35.80417613257833, 
    't_slope_strong': 3.0900059112168865, 
    't_tp_mult': 14.904184250871236,
    't_vol_limit_strong': 0.9932614859957032, 
    't_sl_base_strong': 0.0016337647551746578,
    't_rsi_max_strong': 73.82123798299693, 
    't_rsi_min_strong': 24.697848144515298,

    # -- 공통 추세 파라미터 (트레일링 스탑) --
    't_ts_mult': 0.00030699059555147683, 
    't_sl_activate': 0.010382251184114049
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
    # back02.py evaluate()와 동일한 파라미터 논리 보정 (원본 dict 보호)
    ind = dict(ind)
    # 1) RSI 역전 보정
    if ind['rsi_low'] >= ind['rsi_high']:
        ind['rsi_low'], ind['rsi_high'] = ind['rsi_high'] - 5, ind['rsi_low'] + 5
    # 2) 같은 TF일 때 ADX/slope 역전 보정
    if ind.get('t_inter_normal') == ind.get('t_inter_strong'):
        if ind['t_adx_limit_normal'] > ind['t_adx_limit_strong']:
            ind['t_adx_limit_normal'], ind['t_adx_limit_strong'] = \
                ind['t_adx_limit_strong'], ind['t_adx_limit_normal']
        if ind['t_slope_min'] > ind['t_slope_strong']:
            ind['t_slope_min'], ind['t_slope_strong'] = \
                ind['t_slope_strong'], ind['t_slope_min']

    r_tf  = ind['r_inter']
    tn_tf = ind['t_inter_normal']
    ts_tf = ind['t_inter_strong']
    atr_tf = ind['atr_inter']
    bal, peak_bal, max_dd = 100.0, 100.0, 0.0
    pos = None
    
    # 롱/숏 분리 추적을 위해 딕셔너리 확장
    stats = {
        'range': {'trades': 0, 'wins': 0, 'profit': 0.0, 'long_trades': 0, 'long_wins': 0, 'long_profit': 0.0, 'short_trades': 0, 'short_wins': 0, 'short_profit': 0.0},
        'trend_normal': {'trades': 0, 'wins': 0, 'profit': 0.0, 'long_trades': 0, 'long_wins': 0, 'long_profit': 0.0, 'short_trades': 0, 'short_wins': 0, 'short_profit': 0.0},
        'trend_strong': {'trades': 0, 'wins': 0, 'profit': 0.0, 'long_trades': 0, 'long_wins': 0, 'long_profit': 0.0, 'short_trades': 0, 'short_wins': 0, 'short_profit': 0.0}
    }
    trade_log = []
    trade_returns = []
    pos_duration = 0

    for row in df_main.itertuples():
        curr_p = row.close
        
        if pos is None:
            mode, side = None, None

            if getattr(row, f"adx_{r_tf}") < ind['r_adx_limit'] and \
               getattr(row, f"adx_slope_{r_tf}") <= ind['r_slope_max'] and \
               getattr(row, f"bbw_slope_{r_tf}") < 0:
                if row.vol > (row.vol_mean * ind['r_vol_limit']):
                    ma_v = getattr(row, f"ma20_{r_tf}")
                    
                    # [수정] r_tf(시간봉) 대신 3분봉 RSI인 rsi_3m을 사용합니다.
                    rsi_v = row.rsi_3m 
                    
                    side = 'long'  if (rsi_v < ind['rsi_low']  and curr_p < ma_v) else \
                           'short' if (rsi_v > ind['rsi_high'] and curr_p > ma_v) else None
                    
                    if side: mode = 'range'
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
                else:
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

            if pos['mode'] != 'range':
                entry_p = pos['ent_p']
                if pos['side'] == 'long':
                    current_pnl_pct = (curr_p - entry_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        trailing_sl = curr_p * (1 - ind['t_ts_mult'])
                        if trailing_sl > pos['sl']:
                            pos['sl'] = trailing_sl
                else:
                    current_pnl_pct = (entry_p - curr_p) / (entry_p + 1e-9)
                    if current_pnl_pct > ind['t_sl_activate']:
                        trailing_sl = curr_p * (1 + ind['t_ts_mult'])
                        if trailing_sl < pos['sl']:
                            pos['sl'] = trailing_sl

            if pos['mode'] == 'range' and pos_duration >= 15:
                is_exit = True
                exit_p = curr_p
                
            if not is_exit:
                if (curr_p <= pos['sl'] if pos['side'] == 'long' else curr_p >= pos['sl']):
                    is_exit = True
                    exit_p = pos['sl']
                elif (curr_p >= pos['tp'] if pos['side'] == 'long' else curr_p <= pos['tp']):
                    is_exit = True
                    exit_p = pos['tp']

            if is_exit:
                exit_fee = TAKER_FEE
                effective_bal = bal * 0.1 if bal > BAL_CAP else bal
                pnl = effective_bal * (((exit_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - exit_p)/pos['ent_p']) - (TAKER_FEE + exit_fee + SLIPPAGE*2)) * LEVERAGE
                bal += pnl
                if bal > peak_bal: peak_bal = bal
                max_dd = max(max_dd, (peak_bal - bal) / (peak_bal + 1e-9))
                
                # 통계 업데이트 (롱/숏 분리)
                side_prefix = f"{pos['side']}_" # 'long_' or 'short_'
                stats[pos['mode']]['trades'] += 1
                stats[pos['mode']]['profit'] += pnl
                stats[pos['mode']][side_prefix + 'trades'] += 1
                stats[pos['mode']][side_prefix + 'profit'] += pnl
                
                if pnl > 0: 
                    stats[pos['mode']]['wins'] += 1
                    stats[pos['mode']][side_prefix + 'wins'] += 1

                # 로그에 side 추가
                trade_log.append({'ts': row.ts, 'side': pos['side'], 'profit': pnl, 'bal': bal, 'mode': pos['mode']})
                trade_returns.append(pnl)
                pos = None
                if bal <= 5.0: break

    return bal, trade_log, stats, peak_bal, max_dd, trade_returns

if __name__ == "__main__":
    total_days = TEST_DAYS + BUFFER_DAYS
    df_main = get_data(SYMBOL, '3m', total_days)
    
    if not df_main.empty:
        # [수정된 부분] 중복 제거 및 데이터 전처리 통합
        df_main['vol_mean'] = df_main['vol'].rolling(20).mean()
        df_main['rsi_3m'] = ta.rsi(df_main['close'], length=14)
        
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
        
        print("\n" + "="*70)
        print(f"📊 {SYMBOL} 통합 성과 보고서 (Long/Short 상세 분석)")
        print(f"💰 최종 잔고: ${final_bal:,.2f} | 수익률: {((final_bal-100)/100)*100:,.1f}%")
        print(f"🔝 최고 잔고: ${peak_bal:,.2f} | 📉 최대 낙폭(MDD): {max_dd*100:.2f}%")
        print(f"🤝 전체 거래: {total_trades}회 | 승률: {(total_wins/total_trades*100) if total_trades>0 else 0:.1f}%")

        print("\n" + "─"*22 + " [모드별 상세 성과] " + "─"*25)
        display_map = {'range': '횡보(Range)', 'trend_normal': '일반추세(Normal)', 'trend_strong': '파워추세(Power)'}
        
        abs_total_profit = sum(abs(mode_stats[m]['profit']) for m in all_modes)
        for m in all_modes:
            m_data = mode_stats[m]
            m_winrate = (m_data['wins']/m_data['trades']*100) if m_data['trades'] > 0 else 0
            contribution = (abs(m_data['profit']) / (abs_total_profit + 1e-9) * 100)
            
            # 롱/숏 개별 통계 계산
            l_tr, s_tr = m_data['long_trades'], m_data['short_trades']
            l_winrate = (m_data['long_wins']/l_tr*100) if l_tr > 0 else 0
            s_winrate = (m_data['short_wins']/s_tr*100) if s_tr > 0 else 0
            
            print(f"▶ {display_map[m]:14} | 전체 {m_data['trades']:3}회 | 승률 {m_winrate:5.1f}% | 총수익 ${m_data['profit']:,.2f} (비중 {contribution:4.1f}%)")
            print(f"   ├─ [🔴Long ]   {l_tr:3}회 | 승률 {l_winrate:5.1f}% | 수익 ${m_data['long_profit']:,.2f}")
            print(f"   └─ [🔵Short]   {s_tr:3}회 | 승률 {s_winrate:5.1f}% | 수익 ${m_data['short_profit']:,.2f}")
            print("")

        if logs:
            df_log = pd.DataFrame(logs)
            print("─"*24 + " [월별 수익 추이] " + "─"*27)
            monthly_stats = df_log.set_index('ts')['profit'].resample('ME').sum()
            print(monthly_stats)

            print("\n" + "─"*24 + " [치명적 손실 분석] " + "─"*25)
            losses = df_log[df_log['profit'] < 0]['profit']
            if not losses.empty:
                print(f"📉 평균 손실액: ${abs(losses.mean()):,.2f}")
                print(f"😱 최대 단일 손실: ${abs(losses.min()):,.2f}")

            print(f"\n⚠️ [마지막 거래 기록 (청산 원인 파악)]")
            for last_trade in logs[-3:]:
                side_icon = "🔴L" if last_trade.get('side') == 'long' else "🔵S"
                print(f"   {last_trade['ts']} | [{side_icon}] 수익: ${last_trade['profit']:,.2f} | 잔고: ${last_trade['bal']:,.2f} ({display_map[last_trade['mode']]})")
        print("="*70)
