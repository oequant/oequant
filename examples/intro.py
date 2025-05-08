#%%
# Cell 1: Imports and Data Loading
import oequant as oq
import pandas as pd
import pandas_ta as ta
from bokeh.io import output_notebook
output_notebook() # this is needed for interactive plots


# Load sample data
ticker = "QQQ"
start_date = "2010-01-01"
end_date = "2025-05-01" 

print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
data = oq.get_data(ticker, start=start_date, end=end_date)
print("Data loaded successfully:")
print(data.tail())

#%%
# Cell 2: Feature Engineering
print("\nCalculating features...")
# Calculate RSI_3
data.ta.rsi(length=3, append=True, col_names=('rsi_3',)) # Appends 'RSI_3'

# Calculate exit signal: close > high.shift()
data['exit_signal'] = data['close'] > data['high'].shift(1)

# Entry signal: rsi_3 < 20 (will be passed as an expression)
entry_expression = "rsi_3 < 20"

print("Features calculated:")
print(data[['close', 'high', 'rsi_3', 'exit_signal']].tail())


#%%
# Cell 3: Backtesting
print("\nRunning backtest...")
# Define backtest parameters
initial_capital = 100_000
# Entry: rsi_3 < 20
# Exit: our 'exit_signal' column

print(f"Data shape after NaN drop: {data.shape}")
print(f"Entry expression: {entry_expression}")
print(f"Exit column name: 'exit_signal'")


results = oq.backtest(
    data=data,
    entry_column=entry_expression, # "rsi_3 < 20"
    exit_column='exit_signal',     # Pre-calculated boolean column
    capital=initial_capital,
    fee_frac=0.001 # Example fee
)
print("Backtest completed.")

#%%
# Cell 4: Reporting
print("\nGenerating report...")
results.report()

print("\nScript finished.") 
# %%
