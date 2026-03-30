import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

# =============================================================================
# 1. TRADING CONFIGURATION
# =============================================================================

INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 500_000
# TIME_STOP_DAYS = 45 # Replaced by dynamic 2x Half-Life
EXIT_STOP_LOSS_SIGMA = 3
ENTRY_THRESHOLD_SIGMA = 1
COST_RATE = 0.00075 # 0.075% per turn

# Periods
WARMUP_START = "2024-01-01"
WARMUP_END = "2024-12-31"
TRADING_START = "2025-01-01"
TRADING_END = "2026-03-31"

def calculate_costs(trade_value):
    return COST_RATE * trade_value

# =============================================================================
# 2. KALMAN FILTER ENGINE
# =============================================================================

def run_kalman_filter(stock1_total, stock2_total, dates):
    # Split into warmup for Ve estimation
    stock1_warmup = stock1_total[dates <= WARMUP_END]
    stock2_warmup = stock2_total[dates <= WARMUP_END]
    
    # Estimate Ve using OLS
    x_warmup = sm.add_constant(stock2_warmup.values)
    y_warmup = stock1_warmup.values
    model = sm.OLS(y_warmup, x_warmup).fit()
    ve_estimated = np.var(model.resid)

    # Kalman Parameters
    delta = 0.0001
    vw = (delta / (1 - delta)) * np.eye(2)
    ve = ve_estimated
    p = np.zeros((2, 2))
    beta = np.zeros(2) # [slope, intercept]
    
    results = []
    
    for t in range(len(stock1_total)):
        x_t = np.array([stock2_total.iloc[t], 1.0])
        y_t = stock1_total.iloc[t]
        
        # Prediction step
        if t > 0:
            r = p + vw
        else:
            r = p.copy()
            
        yhat = x_t @ beta
        q_t = x_t @ r @ x_t.T + ve
        e_t = y_t - yhat  # spread
        
        # Update step
        k = (r @ x_t.T) / q_t
        beta = beta + k * e_t
        p = r - np.outer(k, x_t) @ r
        
        period = 'warmup' if dates[t] <= pd.Timestamp(WARMUP_END) else 'trading'
        
        results.append({
            'date': dates[t],
            's1_price': y_t,
            's2_price': stock2_total.iloc[t],
            'hedge_ratio': beta[0],
            'intercept': beta[1],
            'spread': e_t,
            'Q': q_t,
            'sqrt_Q': np.sqrt(q_t),
            'period': period
        })
        
    return pd.DataFrame(results).set_index('date')

# =============================================================================
# 3. BACKTEST ENGINE
# =============================================================================

def run_backtest(df_total, time_stop_days):
    trading_res = df_total[df_total['period'] == 'trading'].copy().reset_index()
    
    position = 0 # 1=long, -1=short
    entry_date = None
    entry_s1 = 0
    entry_s2 = 0
    shares_s1 = 0
    shares_s2 = 0
    entry_beta = 0
    rebal_beta = 0
    entry_cost = 0
    rebalance_count = 0
    rebal_costs_this_trade = 0
    trades = []
    current_capital = INITIAL_CAPITAL

    for t in range(len(trading_res)):
        row = trading_res.iloc[t]
        date = row['date']
        spread = row['spread']
        sqrt_Q = row['sqrt_Q']
        s1_price = row['s1_price']
        s2_price = row['s2_price']
        beta = row['hedge_ratio']

        exit_triggered = False
        exit_type = None

        if position != 0:
            days_held = (date - entry_date).days
            
            # Exit rules
            if (position == 1 and spread < -EXIT_STOP_LOSS_SIGMA * sqrt_Q) or \
               (position == -1 and spread > EXIT_STOP_LOSS_SIGMA * sqrt_Q):
                exit_type = 'stop_loss'
                exit_triggered = True
            elif days_held > time_stop_days:
                exit_type = 'time_stop'
                exit_triggered = True
            elif (position == 1 and spread > 0) or (position == -1 and spread < 0):
                exit_type = 'profit'
                exit_triggered = True
            else:
                # Dynamic rebalancing (if beta changes > 10%)
                beta_change = abs(beta - rebal_beta) / rebal_beta if rebal_beta != 0 else 0
                if beta_change > 0.10:
                    new_shares_s2 = int(np.floor(shares_s1 * beta))
                    diff_shares = new_shares_s2 - shares_s2
                    rebal_cost = calculate_costs(abs(diff_shares) * s2_price)
                    rebal_costs_this_trade += rebal_cost
                    shares_s2 = new_shares_s2
                    rebal_beta = beta
                    rebalance_count += 1
        
        if exit_triggered:
            if position == 1:
                gross_pnl = (shares_s1 * (s1_price - entry_s1)) - (shares_s2 * (s2_price - entry_s2))
            else:
                gross_pnl = (-shares_s1 * (s1_price - entry_s1)) + (shares_s2 * (s2_price - entry_s2))
            
            exit_cost = calculate_costs(shares_s1 * s1_price) + calculate_costs(shares_s2 * s2_price)
            total_trade_costs = entry_cost + exit_cost + rebal_costs_this_trade
            net_pnl = gross_pnl - total_trade_costs
            current_capital += net_pnl
            
            trades.append({
                'direction': 'long' if position == 1 else 'short',
                'net_pnl': net_pnl,
                'holding_days': days_held,
                'exit_type': exit_type
            })
            position = 0
            
        if position == 0:
            if spread < -ENTRY_THRESHOLD_SIGMA * sqrt_Q:
                position = 1
                entry_date, entry_s1, entry_s2 = date, s1_price, s2_price
                entry_beta = rebal_beta = beta
                shares_s1 = int(np.floor(POSITION_SIZE / s1_price))
                shares_s2 = int(np.floor(shares_s1 * beta))
                entry_cost = calculate_costs(shares_s1 * s1_price) + calculate_costs(shares_s2 * s2_price)
                rebal_costs_this_trade = 0
                rebalance_count = 0
            elif spread > ENTRY_THRESHOLD_SIGMA * sqrt_Q:
                position = -1
                entry_date, entry_s1, entry_s2 = date, s1_price, s2_price
                entry_beta = rebal_beta = beta
                shares_s1 = int(np.floor(POSITION_SIZE / s1_price))
                shares_s2 = int(np.floor(shares_s1 * beta))
                entry_cost = calculate_costs(shares_s1 * s1_price) + calculate_costs(shares_s2 * s2_price)
                rebal_costs_this_trade = 0
                rebalance_count = 0
                
    return trades, current_capital

# =============================================================================
# 4. MAIN BATCH PROCESS
# =============================================================================

def main():
    try:
        stable_pairs = pd.read_csv('recently_cointegrated_pairs.csv')
    except FileNotFoundError:
        print("Error: recently_cointegrated_pairs.csv not found.")
        return

    print(f"Loaded {len(stable_pairs)} recently cointegrated pairs for Kalman backtesting.")
    
    summary_results = []
    
    for i, row in stable_pairs.iterrows():
        s1, s2 = row['Stock1'], row['Stock2']
        tickers = [f"{s1}.NS", f"{s2}.NS"]
        print(f"\n[{i+1}/{len(stable_pairs)}] Testing {s1}/{s2}...")
        
        # 4a. Download
        df_prices = yf.download(tickers, start=WARMUP_START, end=TRADING_END, progress=False)['Close']
        df_prices = df_prices.dropna()
        
        if df_prices.empty:
            print(f"Skipping {s1}-{s2}: No data found.")
            continue
            
        # 4b. Kalman
        kalman_df = run_kalman_filter(df_prices[f"{s1}.NS"], df_prices[f"{s2}.NS"], df_prices.index)
        
        # 4c. Backtest (Time Stop = 3.32 * Half-Life)
        pair_half_life = row['Half_Life_Days']
        dynamic_time_stop = round(3.32 * pair_half_life)
        trades, final_cap = run_backtest(kalman_df, dynamic_time_stop)
        
        if not trades:
            summary_results.append({
                'Pair': f"{s1}/{s2}", 'Total_Return%': 0, 'CAGR%': 0, 'Sharpe': 0, 
                'MaxDD%': 0, 'Trades': 0, 'WinRate%': 0, 'Status': 'No trades'
            })
            continue
            
        # 4d. Metrics
        trades_df = pd.DataFrame(trades)
        total_pnl = final_cap - INITIAL_CAPITAL
        total_ret = (total_pnl / INITIAL_CAPITAL) * 100
        
        trading_days = len(kalman_df[kalman_df['period'] == 'trading'])
        cagr = ((final_cap / INITIAL_CAPITAL) ** (252 / trading_days) - 1) * 100
        
        win_rate = (len(trades_df[trades_df['net_pnl'] > 0]) / len(trades_df)) * 100
        
        # Sharpe
        trade_rets = trades_df['net_pnl'] / INITIAL_CAPITAL
        sharpe = (trade_rets.mean() / trade_rets.std() * np.sqrt(252 / trades_df['holding_days'].mean())) if len(trades) > 1 and trade_rets.std() != 0 else 0
        
        # Drawdown
        cum_pnl = trades_df['net_pnl'].cumsum()
        max_dd_pct = (abs((cum_pnl - cum_pnl.cummax()).min()) / INITIAL_CAPITAL) * 100
        
        summary_results.append({
            'Pair': f"{s1}/{s2}",
            'Total_Return%': total_ret,
            'CAGR%': cagr,
            'Sharpe': sharpe,
            'MaxDD%': max_dd_pct,
            'Trades': len(trades),
            'WinRate%': win_rate,
            'Status': 'Completed'
        })
        print(f"Completed: Return {total_ret:.2f}%, CAGR {cagr:.2f}%, Win Rate {win_rate:.1f}%")

    # 5. Output Summary
    summary_df = pd.DataFrame(summary_results).sort_values('Total_Return%', ascending=False)
    
    with open('kalman_backtest_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 105 + "\n")
        f.write(f"      KALMAN FILTER PAIRS TRADING BATCH BACKTEST RESULTS ({TRADING_START[:4]})\n")
        f.write("=" * 105 + "\n\n")
        
        header = f"{'Pair':<22} | {'Return%':>8} | {'CAGR%':>8} | {'Sharpe':>7} | {'MaxDD%':>7} | {'Trades':>6} | {'Win%':>6} | {'Status':<10}\n"
        f.write(header)
        f.write("-" * 105 + "\n")
        
        for _, r in summary_df.iterrows():
            line = f"{r['Pair']:<22} | {r['Total_Return%']:>8.1f} | {r['CAGR%']:>8.1f} | {r['Sharpe']:>7.2f} | {r['MaxDD%']:>7.1f} | {r['Trades']:>6} | {r['WinRate%']:>5.1f}% | {r['Status']:<10}\n"
            f.write(line)
            
        f.write("\n" + "=" * 105 + "\n")
        f.write(f"Parameters: Entry=1σ, Exit=0, StopLoss=3σ, TimeStop=3.32x Half-Life, Rebalancing=10% Beta Change\n")

    print("\nBatch backtest summary saved to kalman_backtest_summary.txt")

if __name__ == "__main__":
    main()
