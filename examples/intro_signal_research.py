#%%
# Cell 1: Imports and Setup
import oequant as oq
import pandas_ta as ta
import pandas as pd # Required for pd.to_datetime in example usage if oos_from is a string

# Import the new research functionalities
from oequant.research.returns import calculate_returns
from oequant.research.signal import research_signal_bins

# For notebook-like environments, to display plotly figures inline if not done by research_signal_bins default
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True) # Uncomment if plots don't show automatically

print("oequant and research modules imported.")

# --- Installation Check for oequant (similar to intro.py) --- #
# This is usually not needed if running from a properly setup environment
# but included for robustness if this script is run standalone.
# try:
#     import oequant as oq
#     print("oequant package already installed.")
# except ImportError:
#     print("oequant package not found. Installing from current directory...")
#     import subprocess
#     import sys
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
#         print("Installation successful!")
#         import oequant as oq # Try importing again
#     except Exception as e:
#         print(f"ERROR: Failed to install oequant: {e}")
#         print("Please ensure oequant is installed and accessible.")
#         sys.exit(1) 
# --- End Installation Check --- #


#%%
# Cell 2: Load Data
ticker = "QQQ"
start_date = "2010-01-01"
end_date = "2025-05-01" # Using a recent end date

print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
dfi = oq.get_data(ticker, start=start_date, end=end_date)
print("Data loaded successfully:")
print(dfi.tail())


#%%
# Cell 3: Calculate Indicators
print("\nCalculating base indicators...")
# Calculate RSI_3
dfi.ta.rsi(length=3, append=True, col_names=('rsi_3',))
print("rsi_3 calculated.")
print(dfi[['close', 'rsi_3']].tail())


#%%
# Cell 4: Calculate Forward Returns
# We need forward returns to see the predictive power of rsi_3 quantiles
print("\nCalculating forward returns...")
# Calculate 1-day forward return on Close price
# The calculate_returns function from oequant.research.returns is used here.
forward_returns_df = calculate_returns(
    dfi.copy(), 
    periods=(1, 5, 20), # Calculate 1-day, 5-day, 20-day forward returns
    forward=True, 
    calculate_ons_ids=False, # Don't need c2o/o2c for this example
    price_column='close',
    price_column_entry='close'
)
dfi = pd.concat([dfi, forward_returns_df], axis=1)

# Let's select one forward return column for the analysis
forward_ret_col_to_analyze = 'forward_return_01' # Using 1-day forward return

print(f"Forward returns calculated. Using '{forward_ret_col_to_analyze}' for analysis.")
print(dfi[['close', 'rsi_3', forward_ret_col_to_analyze]].tail())


#%%
# Cell 5: Research RSI_3 Signal Bins
# Analyze the predictive power of rsi_3 quantiles on forward_return_01
signal_to_analyze = 'rsi_3'
num_quantiles = 10

print(f"\nRunning signal research for '{signal_to_analyze}' split into {num_quantiles} quantiles.")
print(f"Target forward return column: '{forward_ret_col_to_analyze}'.")

# Drop rows where essential columns might be NaN before passing to research function
# research_signal_bins also handles NaNs, but this ensures clean input for the specific columns of interest.
df_cleaned = dfi.dropna(subset=[signal_to_analyze, forward_ret_col_to_analyze])

if df_cleaned.empty:
    print(f"DataFrame is empty after dropping NaNs in '{signal_to_analyze}' or '{forward_ret_col_to_analyze}'. Cannot proceed.")
else:
    results = research_signal_bins(
        df=df_cleaned,
        signal_cols=signal_to_analyze,
        forward_ret_col=forward_ret_col_to_analyze,
        split_kind='qcut',
        split_params=num_quantiles,  # 10 quantiles
        show_stats=True,
        show_pnl_plot=True,
        # oos_from='2022-01-01' # Optional: Uncomment to test OOS splitting
    )

    print("\nSignal research finished.")
    if results:
        print("Returned dictionary keys:", results.keys())
        # The stats DataFrame and Plotly figure object are returned.
        # If not in a notebook, plots might not show automatically but the figure object is available.


#%%
# Cell 6: Example with multiple signals and OOS
print("\nRunning example with multiple signals and OOS splitting...")

# Calculate another indicator for multi-signal analysis
dfi.ta.rsi(length=14, append=True, col_names=('rsi_14',))
dfi.ta.bbands(length=20, append=True, col_names=('bbl_20_2.0', 'bbm_20_2.0', 'bbu_20_2.0', 'bbb_20_2.0', 'bbp_20_2.0'))
dfi['bbp_20_2.0_bin'] = pd.qcut(dfi['bbp_20_2.0'], 5, labels=False, duplicates='drop') # Pre-bin one for demo

signals_to_analyze_multi = ['rsi_3', 'rsi_14'] # research_signal_bins will bin these
# signals_to_analyze_multi = ['rsi_3', 'bbp_20_2.0_bin'] # If one is pre-binned (adjust research_signal_bins usage)

# Ensure the new signals and the forward return column exist and are not all NaN
df_cleaned_multi = dfi.dropna(subset=signals_to_analyze_multi + [forward_ret_col_to_analyze])

if df_cleaned_multi.empty:
    print(f"DataFrame is empty for multi-signal analysis after dropping NaNs. Cannot proceed.")
else:
    results_multi_oos = research_signal_bins(
        df=df_cleaned_multi,
        signal_cols=signals_to_analyze_multi,
        forward_ret_col=forward_ret_col_to_analyze,
        split_kind='qcut',
        split_params=3,  # Use fewer quantiles for multi-signal to keep combinations manageable
        oos_from='2023-01-01',
        show_stats=True,
        show_pnl_plot=True
    )
    print("\nMulti-signal OOS research finished.")

#%%
print("\nIntro signal research script finished.") 