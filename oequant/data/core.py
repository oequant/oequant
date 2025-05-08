import pandas as pd
import yfinance as yf

def get_data(symbols, start=None, end=None, data_source='yf', **kwargs):
    """
    Fetches historical market data.

    Args:
        symbols (str or list): A single symbol string or a list of symbols.
        start (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
        data_source (str, optional): The data source to use. Currently only 'yf' 
                                     (Yahoo Finance) is supported. Defaults to 'yf'.
        **kwargs: Additional arguments to pass to the yfinance.download() function.
                  Example: auto_adjust=True (which is default in newer yfinance)
                           or group_by='column' for different multi-symbol layout.

    Returns:
        pd.DataFrame: A DataFrame with historical data.
                      If a single symbol is provided, columns are [open, high, low, close, volume].
                      Date is the index. All column names are lowercase. 'adj close' is renamed to 'close'.
                      If multiple symbols are provided and default yfinance group_by='column' (or not specified)
                      is used, columns are MultiIndex ('Open', 'QQQ'), ('Close', 'SPY').
                      This function will then stack it to have a MultiIndex (Date, symbol) on rows.
    """
    if data_source.lower() != 'yf':
        raise NotImplementedError(f"Data source '{data_source}' is not yet supported.")

    # Default to auto_adjust=True as yfinance does.
    # yfinance also defaults group_by='column' for multiple tickers if not specified.
    # If user passes group_by='ticker', yf.download structure is different.
    # We will primarily support the yf default group_by='column' and then stack.

    data = yf.download(symbols, start=start, end=end, progress=False, **kwargs)

    if data.empty:
        print(f"No data found for {symbols} between {start} and {end}.")
        return pd.DataFrame() # Return an empty DataFrame

    print(f"Initial yfinance columns: {data.columns}") 
    print(f"Type of yfinance columns: {type(data.columns)}")
    print(f"Is MultiIndex: {isinstance(data.columns, pd.MultiIndex)}")

    if isinstance(data.columns, pd.MultiIndex):
        # Handles multiple symbols when yfinance default group_by='column' is used,
        # or if yfinance returns MultiIndex for a single symbol for some reason.
        # Columns are like: ('Adj Close', 'QQQ'), ('Adj Close', 'SPY'), ('Close', 'QQQ'), ...
        
        new_cols = []
        for col_tuple in data.columns:
            # col_tuple is typically (PriceType, Symbol), e.g., ('Adj Close', 'QQQ')
            price_type = str(col_tuple[0]).lower().replace(' ', '_')
            symbol_level = str(col_tuple[1]) # Keep symbol as is for now
            if price_type == 'adj_close':
                price_type = 'close'
            new_cols.append((symbol_level, price_type)) # Swap order for desired stacking

        data.columns = pd.MultiIndex.from_tuples(new_cols, names=['symbol', 'feature'])
        
        # If only one symbol was requested but returned MultiIndex, level 0 might be the symbol.
        # If multiple symbols, level 0 is symbol.
        # Stacking 'symbol' level will bring it to the index.
        data = data.stack(level='symbol',dropna=False) # stack the 'symbol' level

    else: # Single symbol, or yfinance returned flat columns for multiple symbols (e.g. group_by='row')
          # This path expects flat column names like 'Open', 'Adj Close' etc.
        
        new_cols = []
        for col in data.columns:
            col_name = str(col) # Ensure it's a string
            # Handle cases where yfinance might return ('Close',) for a single column
            if isinstance(col, tuple) and len(col) == 1:
                col_name = str(col[0])
            
            col_name = col_name.lower().replace(' ', '_')
            if col_name == 'adj_close':
                col_name = 'close'
            new_cols.append(col_name)
        data.columns = new_cols
        
        # If a single symbol string was passed, add it as a 'symbol' column for consistency, then set index
        if isinstance(symbols, str):
            data['symbol'] = symbols
            if isinstance(data.index, pd.DatetimeIndex):
                 # If Date is already index, add symbol to make MultiIndex
                data = data.set_index(['symbol'], append=True)


    # Ensure index names are lowercase
    if isinstance(data.index, pd.MultiIndex):
        data.index.names = [name.lower() if name else None for name in data.index.names]
    elif data.index.name:
        data.index.name = data.index.name.lower()
    
    # If 'date' is not already part of index name, ensure first level of index is 'date'
    if isinstance(data.index, pd.MultiIndex):
        current_names = list(data.index.names)
        if 'date' not in [n.lower() for n in current_names if n]:
            # Assuming the first level is the date index from yfinance
            if current_names[0] is None or current_names[0].lower() != 'date':
                 current_names[0] = 'date'
            data.index.names = current_names
    elif data.index.name is None or data.index.name.lower() != 'date': # Single index
        data.index.name = 'date'


    # Standardize to (Date, Symbol) index if not already
    if 'symbol' in data.columns and isinstance(data.index, pd.DatetimeIndex) and data.index.name == 'date':
        data = data.set_index('symbol', append=True)
    
    # Reorder index levels if 'symbol' and 'date' are present, to make it (date, symbol)
    if isinstance(data.index, pd.MultiIndex) and 'date' in data.index.names and 'symbol' in data.index.names:
        if data.index.names.index('date') != 0 or data.index.names.index('symbol') != 1:
            data = data.reorder_levels(['date', 'symbol'])
            
    # If after all processing, we have a single symbol and it's in the index,
    # for a single symbol query, users might prefer a simple DatetimeIndex.
    # However, the backtester might expect 'symbol' to be available.
    # For now, keep (Date, Symbol) multi-index or (Date) index with 'symbol' column.
    # The example script uses a single symbol. Backtester expects data[entry_column] etc.
    # If single symbol, remove the symbol index level to match original design for single ticker.
    if isinstance(symbols, str) and isinstance(data.index, pd.MultiIndex) and 'symbol' in data.index.names:
        symbol_val = data.index.get_level_values('symbol').unique()
        if len(symbol_val) == 1 and symbol_val[0] == symbols:
            data = data.droplevel('symbol')
            # The 'symbol' column might have been added and then removed from index.
            # Ensure it's not a redundant column if index is now just Date.
            if 'symbol' in data.columns:
                del data['symbol']

    print(f"Processed data columns: {data.columns}")
    print(f"Processed data index: {data.index}")
    return data 