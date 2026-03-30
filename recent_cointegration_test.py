import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import itertools
import os

# Global date range for both data download and testing window
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# =============================================================================
# 1. SETUP & DATA DOWNLOAD
# =============================================================================

def download_all_data(pairs):
    all_tickers = list(set(
        [p[0] + '.NS' for p in pairs] + 
        [p[1] + '.NS' for p in pairs]
    ))
    
    print(f"Downloading {len(all_tickers)} unique tickers for recent check from {START_DATE} to {END_DATE}...")
    
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)['Close']
    log_prices = np.log(raw)
    return log_prices

# =============================================================================
# 2. STATISTICAL FUNCTIONS
# =============================================================================

def calculate_half_life(spread):
    try:
        delta_spread = spread.diff().dropna()
        lagged_spread = spread.shift(1).dropna()
        
        # Align
        common_idx = delta_spread.index.intersection(lagged_spread.index)
        y = delta_spread.loc[common_idx]
        x = lagged_spread.loc[common_idx]
        
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        lambda_coeff = model.params.iloc[1]
        
        if lambda_coeff >= 0:
            return np.inf
        
        return -np.log(2) / lambda_coeff
    except:
        return np.inf

# =============================================================================
# 3. RECENT WINDOW STABILITY PIPELINE (Last 6 Months)
# =============================================================================

def main():
    # Load screening results
    try:
        screening_results = pd.read_csv('pair_screening_results.csv')
    except FileNotFoundError:
        print("Error: pair_screening_results.csv not found.")
        return

    pairs_list = list(zip(screening_results['Stock1'], screening_results['Stock2']))
    sector_map = dict(zip(zip(screening_results['Stock1'], screening_results['Stock2']), screening_results['Sector']))
    
    log_prices = download_all_data(pairs_list)
    
    final_stability_summary = []
    
    print(f"\nAnalyzing Recent Cointegration for Window: {START_DATE} to {END_DATE}")
    print("-" * 115)
    
    for pair in pairs_list:
        s1, s2 = pair
        s1_t, s2_t = s1 + ".NS", s2 + ".NS"
        
        try:
            # Slice data for the recent window
            data_window = log_prices[[s1_t, s2_t]].loc[START_DATE:END_DATE].dropna()
            
            if len(data_window) < 60:
                print(f"Skipping {s1}-{s2}: Insufficient data ({len(data_window)} points).")
                continue
            
            # Johansen
            jres = coint_johansen(data_window, det_order=0, k_ar_diff=1)
            trace_stat = jres.lr1[0]
            trace_crit = jres.cvt[0, 1]
            trace_pass = trace_stat > trace_crit
            
            # Hedge Ratio
            coint_vector = jres.evec[:, 0]
            if coint_vector[0] < 0:
                coint_vector = -coint_vector
            hedge_ratio = -coint_vector[1] / coint_vector[0]
            
            # Spread and Half-Life
            spread = data_window[s1_t] - hedge_ratio * data_window[s2_t]
            hl = calculate_half_life(spread)
            
            # Hedge Ratio Constraint
            hedge_pass = 0.1 <= hedge_ratio <= 10
            
            # Stability Verdict
            # STABLE if it passes Johansen Trace > Critical @ 5% AND Hedge Ratio in [0.1, 10]
            verdict = "STABLE" if (trace_pass and hedge_pass) else "UNSTABLE"
            
            res = {
                "Pair": f"{s1}-{s2}",
                "Stock1": s1,
                "Stock2": s2,
                "Sector": sector_map[(s1, s2)],
                "Trace_Stat": trace_stat,
                "Trace_Crit": trace_crit,
                "Trace_Pass": trace_pass,
                "Hedge_Ratio": hedge_ratio,
                "Hedge_Pass": hedge_pass,
                "Half_Life": hl,
                "Verdict": verdict
            }
            final_stability_summary.append(res)
            
            # Display results for each pair
            print(f"{s1}-{s2:<25} | Trace: {trace_stat:>7.2f} (Crit: {trace_crit:>7.2f}) | Pass: {str(trace_pass):<5} | HR: {hedge_ratio:>7.4f} (Pass: {str(hedge_pass):<5}) | HL: {hl:>6.2f} | Verdict: {verdict}")
            
        except Exception as e:
            print(f"Error processing {s1}-{s2}: {e}")

    # --- FINAL SUMMARY TABLE ---
    summary_df = pd.DataFrame(final_stability_summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Half_Life")
        print("\n" + "#"*100)
        print("FINAL RECENT WINDOW COINTEGRATION SUMMARY (Jul-Dec 2024)")
        print("#"*100)
        print(summary_df[["Pair", "Sector", "Trace_Pass", "Hedge_Ratio", "Half_Life", "Verdict"]].to_string(index=False))

        # Save CSVs
        summary_df.to_csv("recent_cointegration_results.csv", index=False)
        
        # Stable pairs CSV
        stable_pairs_df = summary_df[summary_df['Verdict'] == "STABLE"].copy()
        
        # Merge with original screening info
        # We rename the recent stats to distinguish from long-term screening results
        out_stable = screening_results.merge(
            stable_pairs_df[["Stock1", "Stock2", "Hedge_Ratio", "Half_Life", "Verdict"]], 
            on=["Stock1", "Stock2"],
            suffixes=('', '_Recent')
        )
        out_stable.to_csv("recently_cointegrated_pairs.csv", index=False)
        
        print(f"\nRecent window results saved to recent_cointegration_results.csv")
        print(f"Recently cointegrated pairs saved to recently_cointegrated_pairs.csv (Count: {len(out_stable)})")

        # Save to TXT
        save_stability_results_to_txt(summary_df)
    else:
        print("No pairs were successfully analyzed.")

def save_stability_results_to_txt(summary_df):
    """Saves the stability results for the recent window to a formatted text file with separate tables."""
    with open("recent_cointegration_results.txt", "w") as f:
        f.write("=" * 115 + "\n")
        f.write(f"      PAIR RECENT WINDOW COINTEGRATION CHECK SUMMARY ({START_DATE} to {END_DATE})\n")
        f.write("=" * 115 + "\n\n")
        
        # Helper to write table
        def write_table(df, title):
            f.write(f"--- {title} ---\n")
            header = f"{'Pair':<22} | {'Sector':<12} | {'Trace Stat':<10} | {'Trace Crit':<10} | {'Pass':<6} | {'Hedge Ratio':<12} | {'Half-Life':<10} | {'Verdict':<10}\n"
            f.write(header)
            f.write("-" * 115 + "\n")
            for _, row in df.iterrows():
                line = f"{row['Pair']:<22} | {row['Sector']:<12} | {row['Trace_Stat']:<10.2f} | {row['Trace_Crit']:<10.2f} | {str(row['Trace_Pass']):<6} | {row['Hedge_Ratio']:<12.4f} | {row['Half_Life']:<10.2f} | {row['Verdict']:<10}\n"
                f.write(line)
            f.write("\n")

        # Split and write
        stable_df = summary_df[summary_df['Verdict'] == "STABLE"].sort_values("Half_Life")
        unstable_df = summary_df[summary_df['Verdict'] == "UNSTABLE"].sort_values("Half_Life")

        if not stable_df.empty:
            write_table(stable_df, "RECENTLY COINTEGRATED PAIRS (Passing all checks)")
        else:
            f.write("No stable pairs found.\n\n")

        if not unstable_df.empty:
            write_table(unstable_df, "UNSTABLE PAIRS")
        else:
            f.write("No unstable pairs found.\n\n")
            
        f.write("=" * 115 + "\n")
        f.write(f"SUMMARY: {len(summary_df)} pairs tested, {len(stable_df)} recently cointegrated pairs found.\n")
        f.write("Verdict criteria: Johansen Trace > Critical @ 5% AND Hedge Ratio in [0.1, 10].\n")

    print("Recent cointegration results successfully saved to recent_cointegration_results.txt")

if __name__ == "__main__":
    main()
