import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import itertools
import time
import os

# =============================================================================
# 1. CONFIGURATION & SECTORS
# =============================================================================

# Define the universe (HDFCBANK, ADANIENT, ADANIPORTS excluded)
SECTORS = {
    "Banking": [
        "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV",
        "FEDERALBNK", "BANDHANBNK", "IDFCFIRSTB", "PNB", "BANKBARODA", "CANARABANK", 
        "UNIONBANK", "MUTHOOTFIN", "CHOLAFIN"
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", 
        "COFORGE", "LTTS", "OFSS", "KPITTECH", "TATAELXSI"
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA", 
        "TORNTPHARM", "BIOCON", "ALKEM", "IPCALAB", "GLENMARK"
    ],
    "Healthcare": [
        "APOLLOHOSP", "MAXHEALTH", "FORTIS", "METROPOLIS", "LALPATHLAB"
    ],
    "Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "BALKRISIND", "MOTHERSON", "BHARATFORG", "ESCORTS", "ASHOKLEY",
        "TVSMOTOR", "APOLLOTYRE"
    ],
    "Energy": [
        "RELIANCE", "ONGC", "BPCL", "POWERGRID", "NTPC", "COALINDIA",
        "IOC", "HINDPETRO", "GAIL", "PETRONET", "OIL", "NHPC", "SJVN"
    ],
    "Metals": [
        "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "SAIL", "NMDC", 
        "NATIONALUM", "RATNAMANI", "JINDALSTEL"
    ],
    "Infra": [
        "LT", "ULTRACEMCO", "GRASIM", "SHREECEM", "SIEMENS", "ABB", 
        "CUMMINSIND", "KEC", "KALPATPOWR", "JKCEMENT", "AMBUJACEM", "ACC"
    ],
    "FMCG": [
        "HINDUNILVR", "NESTLEIND", "BRITANNIA", "ITC", "GODREJCP",
        "DABUR", "MARICO", "COLPAL", "EMAMILTD", "TATACONSUM", "RADICO"
    ]
}

# Auto-correction for TVSMOTOR typo (User wrote TVSMOTOR but APOLLYRE?)
# User wrote: TVSMOTOR, APOLLOTYRE (checking prompt again)
# User wrote: TVSMOTOR, APOLLOTYRE for Auto.
# User wrote: APOLLOTYRE. I typed APOLLYRE. Let me double check stocks.

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# =============================================================================
# 2. DATA DOWNLOAD & PREPROCESSING
# =============================================================================

def download_data():
    all_stocks = []
    for sector_stocks in SECTORS.values():
        all_stocks.extend(sector_stocks)
    all_stocks = sorted(list(set(all_stocks)))
    
    print(f"Starting download for {len(all_stocks)} potential stocks...")
    
    price_data = {}
    dropped_stocks = []
    
    for sector, stocks in SECTORS.items():
        print(f"Downloading sector: {sector}...")
        tickers = [f"{s}.NS" for s in stocks]
        
        # Download one sector at a time to handle per-sector sleep and 5% drop logic easier
        data = yf.download(tickers, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)['Close']
        
        for ticker in tickers:
            symbol = ticker.replace(".NS", "")
            if ticker not in data or data[ticker].isnull().mean() > 0.05:
                dropped_stocks.append(symbol)
            else:
                price_data[symbol] = data[ticker]
        
        time.sleep(0.5)
    
    all_data = pd.DataFrame(price_data)
    # Convert to log prices immediately
    log_data = np.log(all_data)
    
    print(f"\nStocks dropped (>5% missing data): {', '.join(dropped_stocks) if dropped_stocks else 'None'}")
    return log_data, dropped_stocks

# =============================================================================
# 3. STATISTICAL SCREENING FUNCTIONS
# =============================================================================

def calculate_half_life(spread):
    """Calculate Half-life using Ornstein-Uhlenbeck process."""
    try:
        delta_spread = spread.diff().dropna()
        lagged_spread = spread.shift(1).dropna()
        
        # Align
        idx = delta_spread.index.intersection(lagged_spread.index)
        y = delta_spread.loc[idx]
        x = lagged_spread.loc[idx]
        
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        
        lambda_coeff = model.params.iloc[1]
        
        if lambda_coeff >= 0:
            return np.inf
        
        half_life = -np.log(2) / lambda_coeff
        return half_life
    except:
        return np.inf

def screen_sector_pairs(sector_name, stocks, log_data):
    results = []
    sector_pairs_tested = 0
    passed_johansen = 0
    failed_correlation = 0
    failed_hedge_bounds = 0
    passed_half_life = 0
    
    # Filter available stocks
    available_stocks = [s for s in stocks if s in log_data.columns]
    
    pairs = list(itertools.combinations(available_stocks, 2))
    
    for s1, s2 in pairs:
        sector_pairs_tested += 1
        try:
            # Prepare data
            pair_data = log_data[[s1, s2]].dropna()
            
            if len(pair_data) < 60:
                continue
            
            # Step 0: Correlation Filter
            correlation = pair_data[s1].corr(pair_data[s2])
            if correlation < 0.80:
                failed_correlation += 1
                continue

            # Step 1: Johansen Cointegration
            # det_order=0 (constant), k_ar_diff=1 (lag 1)
            jres = coint_johansen(pair_data, det_order=0, k_ar_diff=1)
            
            trace_stat = jres.lr1[0]
            trace_crit = jres.cvt[0, 1] # 95% critical value
            
            if trace_stat <= trace_crit:
                continue
            
            passed_johansen += 1
            
            # Step 2: Hedge Ratio from Johansen
            coint_vector = jres.evec[:, 0]
            
            # Normalize sign
            if coint_vector[0] < 0:
                coint_vector = -coint_vector
            
            hedge_ratio = -coint_vector[1] / coint_vector[0]
            
            if hedge_ratio < 0.1 or hedge_ratio > 10:
                failed_hedge_bounds += 1
                continue
                
            # Step 3: Half-Life
            spread = pair_data[s1] - hedge_ratio * pair_data[s2]
            half_life = calculate_half_life(spread)
            
            if 5 <= half_life <= 30:
                passed_half_life += 1
                results.append({
                    "Sector": sector_name,
                    "Stock1": s1,
                    "Stock2": s2,
                    "Correlation": correlation,
                    "Trace_Stat": trace_stat,
                    "Trace_Crit": trace_crit,
                    "Hedge_Ratio": hedge_ratio,
                    "Half_Life_Days": half_life
                })
                
        except Exception as e:
            # Silent logging as per rules
            continue
            
    return results, sector_pairs_tested, passed_johansen, failed_correlation, failed_hedge_bounds, passed_half_life

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def main():
    log_data, dropped_stocks = download_data()
    
    all_results = []
    total_pairs_tested = 0
    total_passed_johansen = 0
    total_failed_correlation = 0
    total_failed_hedge = 0
    total_passed_half_life = 0
    
    print("\nStarting screening pipeline...")
    
    for sector_name, stocks in SECTORS.items():
        print(f"Testing Sector: {sector_name}")
        sec_results, p_tested, p_johan, f_corr, f_hedge, p_hl = screen_sector_pairs(sector_name, stocks, log_data)
        
        total_pairs_tested += p_tested
        total_passed_johansen += p_johan
        total_failed_correlation += f_corr
        total_failed_hedge += f_hedge
        total_passed_half_life += p_hl
        
        if sec_results:
            # Sort by half-life
            sec_results_df = pd.DataFrame(sec_results).sort_values("Half_Life_Days")
            all_results.extend(sec_results)
            
            print(f"\nResults for {sector_name}:")
            print("-" * 95)
            headers = ["Rank", "Stock1", "Stock2", "Corr", "Trace_Stat", "Trace_Crit", "Hedge_Ratio", "Half_Life_Days"]
            print(f"{headers[0]:<5} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<8} | {headers[4]:<12} | {headers[5]:<12} | {headers[6]:<12} | {headers[7]:<14}")
            print("-" * 105)
            for idx, row in sec_results_df.iterrows():
                rank = list(sec_results_df.index).index(idx) + 1
                print(f"{rank:<5} | {row['Stock1']:<12} | {row['Stock2']:<12} | {row['Correlation']:<8.4f} | {row['Trace_Stat']:<12.4f} | {row['Trace_Crit']:<12.4f} | {row['Hedge_Ratio']:<12.4f} | {row['Half_Life_Days']:<14.2f}")
            print("\n")
        else:
            print(f"No pairs passed filters in {sector_name}.\n")

    # Overall Top 10
    top_10_df = pd.DataFrame(all_results)
    if not top_10_df.empty:
        top_10_df = top_10_df.sort_values("Half_Life_Days").head(10)
        
        print("-" * 30 + " FINAL TOP 10 PAIRS OVERALL " + "-" * 30)
        headers = ["Rank", "Stock1", "Stock2", "Trace_Stat", "Trace_Crit", "Hedge_Ratio", "Half_Life_Days", "Sector"]
        print(f"{headers[0]:<5} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<12} | {headers[6]:<14} | {headers[7]:<12}")
        print("-" * 100)
        for i, (idx, row) in enumerate(top_10_df.iterrows(), 1):
            print(f"{i:<5} | {row['Stock1']:<12} | {row['Stock2']:<12} | {row['Trace_Stat']:<10.2f} | {row['Trace_Crit']:<10.2f} | {row['Hedge_Ratio']:<12.4f} | {row['Half_Life_Days']:<14.2f} | {row['Sector']:<12}")
        print("\n")
        
        # Save to CSV
        full_results_df = pd.DataFrame(all_results).sort_values("Half_Life_Days")
        full_results_df.to_csv("pair_screening_results.csv", index=False)
        print(f"Full results saved to pair_screening_results.csv")

        # Save to TXT
        save_results_to_txt(full_results_df)
    else:
        print("No pairs passed the screening pipeline.")

    # Final Summary
    total_stocks = sum(len(s) for s in SECTORS.values())
    final_pairs = len(all_results)
    
    print("\n" + "="*40)
    print("FINAL SUMMARY COUNTS")
    print("="*40)
    print(f"Total stocks in universe  : {total_stocks}")
    print(f"Stocks dropped            : {len(dropped_stocks)} ({', '.join(dropped_stocks)})")
    print(f"Total pairs tested        : {total_pairs_tested}")
    print(f"Failed correlation < 0.80 : {total_failed_correlation}")
    print(f"Passed Johansen           : {total_passed_johansen}")
    print(f"Failed hedge ratio bounds : {total_failed_hedge}")
    print(f"Passed half-life filter   : {total_passed_half_life}")
    print(f"Final pairs in output     : {final_pairs}")
    print("="*40)

def save_results_to_txt(df):
    """Saves the screening results to a formatted text file."""
    with open("pair_screening_results.txt", "w") as f:
        f.write("=" * 105 + "\n")
        f.write("      NIFTY SECTOR MEAN-REVERSION SCREENING RESULTS (2021-2024)\n")
        f.write("=" * 105 + "\n\n")

        # Total Summary Counts
        f.write("FINAL SUMMARY:\n")
        f.write(f"Total passing pairs identified: {len(df)}\n")
        f.write("-" * 105 + "\n\n")

        # Sector-wise tables
        sectors = df['Sector'].unique()
        for sector in sectors:
            f.write(f"SECTOR: {sector}\n")
            f.write("-" * 105 + "\n")
            
            sec_df = df[df['Sector'] == sector].sort_values("Half_Life_Days")
            
            # Header
            header = f"{'Rank':<5} | {'Stock 1':<12} | {'Stock 2':<12} | {'Trace Stat':<12} | {'Trace Crit':<12} | {'Hedge Ratio':<12} | {'Half-Life (Days)':<16}\n"
            f.write(header)
            f.write("-" * 105 + "\n")
            
            for i, (idx, row) in enumerate(sec_df.iterrows(), 1):
                line = f"{i:<5} | {row['Stock1']:<12} | {row['Stock2']:<12} | {row['Trace_Stat']:<12.4f} | {row['Trace_Crit']:<12.4f} | {row['Hedge_Ratio']:<12.4f} | {row['Half_Life_Days']:<16.2f}\n"
                f.write(line)
            f.write("\n\n")

        # Overall Top 10
        f.write("=" * 105 + "\n")
        f.write("      TOP 10 PAIRS OVERALL (BY LOWEST HALF-LIFE)\n")
        f.write("=" * 105 + "\n\n")
        
        top_10 = df.sort_values("Half_Life_Days").head(10)
        header = f"{'Rank':<5} | {'Stock 1':<12} | {'Stock 2':<12} | {'Sector':<12} | {'Hedge Ratio':<12} | {'Half-Life (Days)':<16}\n"
        f.write(header)
        f.write("-" * 105 + "\n")
        for i, (idx, row) in enumerate(top_10.iterrows(), 1):
            line = f"{i:<5} | {row['Stock1']:<12} | {row['Stock2']:<12} | {row['Sector']:<12} | {row['Hedge_Ratio']:<12.4f} | {row['Half_Life_Days']:<16.2f}\n"
            f.write(line)

    print("Results successfully saved to pair_screening_results.txt")

if __name__ == "__main__":
    main()
