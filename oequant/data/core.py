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

    Returns:
        pd.DataFrame: A DataFrame with historical data.
                      If a single symbol is provided, columns are [open, high, low, close, volume, dividends, stock splits].
                      Date is the index. All column names are lowercase.
                      If multiple symbols are provided, the DataFrame is stacked with a MultiIndex (Date, symbol).
    """
    if data_source.lower() != 'yf':
        raise NotImplementedError(f"Data source '{data_source}' is not yet supported.")

    if isinstance(symbols, str):
        data = yf.download(symbols, start=start, end=end, progress=False, **kwargs)
        if data.empty:
            # Return an empty DataFrame with expected columns if no data is found
            # yfinance typically returns Adj Close, so we'll rename it to close
            # and ensure other standard columns are present if needed.
            # For now, an empty DF from yf.download is fine.
            pass
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        # Ensure standard columns are present, yfinance might vary
        # For simplicity now, we rely on yfinance's output and just lowercase
        # We'll adjust 'adj_close' to 'close' as it's commonly used
        if 'adj_close' in data.columns:
            data = data.rename(columns={'adj_close': 'close'})
        
        # Select and reorder to a standard OHLCV format if desired,
        # for now, we keep all columns from yfinance and just ensure lowercase.
        # standard_cols = ['open', 'high', 'low', 'close', 'volume']
        # present_cols = [col for col in standard_cols if col in data.columns]
        # data = data[present_cols]


    elif isinstance(symbols, list):
        data = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker', **kwargs)
        if data.empty:
            # For multiple symbols, yf.download with group_by='ticker' returns a multi-index column
            # If empty, we'll just return the empty dataframe.
            pass
        
        # When group_by='ticker', columns are MultiIndex: (Symbol, PriceType e.g. Open)
        # We want to stack this to have (Date, Symbol) as index
        
        # First, lowercase the second level of column names (PriceType)
        new_cols = []
        for col_L1, col_L2 in data.columns:
            new_cols.append((col_L1, col_L2.lower().replace(' ', '_')))
        data.columns = pd.MultiIndex.from_tuples(new_cols)

        # Rename 'adj_close' to 'close' at the second level of columns
        renamed_cols = []
        for col_L1, col_L2 in data.columns:
            if col_L2 == 'adj_close':
                renamed_cols.append((col_L1, 'close'))
            else:
                renamed_cols.append((col_L1, col_L2))
        data.columns = pd.MultiIndex.from_tuples(renamed_cols)

        # Stack the DataFrame to bring symbols from columns to index
        data = data.stack(level=0).rename_axis(['Date', 'symbol'])
        
        # Reorder columns to place symbol next to Date if desired, or just keep as is
        # data = data.reset_index().set_index(['Date', 'symbol']) # if symbol was level 1 after stack
        # Ensure columns are in a consistent order if needed
        # For example: data = data[['open', 'high', 'low', 'close', 'volume']]
        # For now, keep all lowercase columns provided by yfinance after stacking.

    else:
        raise TypeError("Symbols must be a string or a list of strings.")

    data.index.name = data.index.name.lower() if data.index.name else None # Ensure index name is lowercase
    if isinstance(data.index, pd.MultiIndex):
        data.index.names = [name.lower() if name else None for name in data.index.names]


    return data 