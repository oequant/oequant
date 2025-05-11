import pandas as pd
import numpy as np
import plotly.express as px
from tabulate import tabulate
# Attempt to import IPython display functionalities for richer output in notebooks
try:
    from IPython.display import display, HTML
    def _is_notebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell': # Jupyter notebook or qtconsole
                return True
            elif shell == 'TerminalInteractiveShell': # Terminal IPython
                return False
            else: # Other type (?)
                return False
        except NameError:
            return False # Probably standard Python interpreter
except ImportError:
    def _is_notebook():
        return False # IPython not available
    def display(obj): # Fallback display
        print(obj)
    def HTML(obj_str): # Fallback HTML
        return obj_str

def _calculate_group_stats(series: pd.Series, ann_factor: int = 252):
    """Calculates performance statistics for a series of returns."""
    series = series.dropna()
    if len(series) == 0:
        return pd.Series({
            'CAGR %': 0, 'Return mean %': 0, 'Return std %': 0,
            'Sharpe Ratio': 0, 'Count': 0, 'Max DD %': 0,
            'Win Rate': 0, 'Avg Win': 0, 'Avg Loss': 0,
            'Profit Factor': np.nan
        })

    count = len(series)
    total_ret = series.sum()
    mean_ret = series.mean()
    std_dev = series.std()
    
    # Calculate CAGR if we have enough data
    if count > 1:
        cagr = (1 + total_ret) ** (ann_factor / count) - 1
    else:
        cagr = total_ret  # For single data point, just use the return
    
    sharpe_ratio = (mean_ret / std_dev) * np.sqrt(ann_factor) if std_dev != 0 and not np.isnan(std_dev) else 0
    
    positive_rets = series[series > 0]
    negative_rets = series[series < 0]

    win_rate = len(positive_rets) / count if count > 0 else 0
    avg_win = positive_rets.mean() if len(positive_rets) > 0 else 0
    avg_loss = negative_rets.mean() if len(negative_rets) > 0 else 0 # This will be negative

    total_gains = positive_rets.sum()
    total_losses = abs(negative_rets.sum()) # abs to make it positive for ratio
    profit_factor = total_gains / total_losses if total_losses != 0 else np.inf
    
    # Max Drawdown
    cumulative_ret = (1 + series).cumprod()
    peak = cumulative_ret.cummax()
    drawdown = (cumulative_ret - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0

    return pd.Series({
        'CAGR %': cagr * 100,
        'Return mean %': mean_ret * 100,
        'Return std %': std_dev * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Count': count,
        'Max DD %': max_drawdown * 100,
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': profit_factor
    })

def research_signal_bins(
    df: pd.DataFrame,
    signal_cols: list[str] | str,
    forward_ret_col: str,
    split_kind: str = 'qcut',
    split_params: int | list = 10,
    oos_from: str | pd.Timestamp | None = None,
    show_stats: bool = True,
    show_pnl_plot: bool = True,
    pivot_aggfunc: str | list | dict = 'mean',
    date_col: str | None = None,
    show_oos_separated: bool = False
):
    """
    Researches signals by splitting them into bins (quantiles or custom) and analyzing forward returns.

    Args:
        df (pd.DataFrame): DataFrame containing signal and return columns. Must have a DatetimeIndex or a date_col.
        signal_cols (list[str] | str): Column name(s) of the signal(s) to be binned.
        forward_ret_col (str): Column name of the forward returns to analyze.
        split_kind (str, optional): 'qcut' for quantiles or 'cut' for custom bins. Defaults to 'qcut'.
        split_params (int | list, optional): 
            For 'qcut', number of quantiles (int, default 10).
            For 'cut', list of bin edges.
        oos_from (str | pd.Timestamp | None, optional): Date to start out-of-sample period. 
            If provided, an 'is_oos' column is added and used in pivot. Defaults to None.
        show_stats (bool, optional): Whether to display the statistics table. Defaults to True.
        show_pnl_plot (bool, optional): Whether to display the cumulative P&L plot. Defaults to True.
        pivot_aggfunc (str | list | dict, optional): Aggregation function for pivot table. Defaults to 'mean'.
        date_col (str | None, optional): Name of the date column if df doesn't have a DatetimeIndex.
                                        If None, df.index is assumed to be DatetimeIndex.
        show_oos_separated (bool, optional): Whether to show in-sample and out-of-sample results separately. 
                                            Only used when oos_from is not None. Defaults to False.

    Returns:
        dict: A dictionary containing 'stats_df' (pd.DataFrame) and 'pnl_fig' (plotly.graph_objects.Figure).
              If show_oos_separated=True and oos_from is provided, returns a dict with 'in_sample' and 'out_sample' keys,
              each containing its own 'stats_df' and 'pnl_fig'.
    """
    if not isinstance(signal_cols, list):
        signal_cols = [signal_cols]

    df_analysis = df.copy()

    if date_col:
        if date_col not in df_analysis.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
        df_analysis = df_analysis.set_index(date_col, drop=True) # Changed drop=False to drop=True
    
    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex, or 'date_col' must be provided and be datetime-like.")

    active_binned_signal_cols = []
    for col in signal_cols:
        binned_col_name = f"{col}_bin"

        if df_analysis[col].isna().all():
            print(f"Warning: Signal column '{col}' is all NaNs. Skipping binning for this column.")
            continue
            
        try:
            binned_series = None
            if split_kind == 'qcut':
                binned_series = pd.qcut(df_analysis[col], q=split_params, labels=False, duplicates='drop')
            elif split_kind == 'cut':
                binned_series = pd.cut(df_analysis[col], bins=split_params, labels=False, duplicates='drop', include_lowest=True)
            # No else needed as split_kind is validated by now or earlier checks should exist
            
            if binned_series is None: # Should not happen if split_kind is valid
                print(f"Warning: Binned series for column '{col}' was not generated. Skipping.")
                continue

            if binned_series.isna().all():
                print(f"Warning: Binning column '{col}' (as '{binned_col_name}') resulted in all NaNs. Skipping this signal.")
                continue
            
            df_analysis[binned_col_name] = binned_series
            active_binned_signal_cols.append(binned_col_name)

        except Exception as e:
            print(f"Warning: Could not {split_kind} column '{col}' (to be '{binned_col_name}'). Skipping. Error: {e}")
            # Ensure a potentially partially created all-NaN column is not left if error happened after assignment and before check
            if binned_col_name in df_analysis.columns and df_analysis[binned_col_name].isna().all():
                df_analysis.drop(columns=[binned_col_name], inplace=True)
            continue

    if not active_binned_signal_cols:
        print("No signal columns were successfully binned and usable. Exiting.")
        return {"stats_df": pd.DataFrame(), "pnl_fig": None}

    # If show_oos_separated is True and oos_from is specified, we'll process in-sample and out-of-sample separately
    if show_oos_separated and oos_from:
        oos_timestamp = pd.to_datetime(oos_from)
        # Split data into in-sample and out-of-sample
        df_in_sample = df_analysis[df_analysis.index < oos_timestamp]
        df_out_sample = df_analysis[df_analysis.index >= oos_timestamp]
        
        if df_in_sample.empty or df_out_sample.empty:
            print("Warning: Either in-sample or out-of-sample dataset is empty. Continuing with combined analysis.")
            show_oos_separated = False
        else:
            results = {
                'in_sample': _process_bins_data(df_in_sample, active_binned_signal_cols, forward_ret_col, pivot_aggfunc, 
                                              "In-Sample", show_stats, show_pnl_plot),
                'out_sample': _process_bins_data(df_out_sample, active_binned_signal_cols, forward_ret_col, pivot_aggfunc, 
                                               "Out-of-Sample", show_stats, show_pnl_plot)
            }
            return results
    
    # Standard processing (either show_oos_separated=False or oos_from is None)
    pivot_group_cols = active_binned_signal_cols
    if oos_from and not show_oos_separated:
        oos_timestamp = pd.to_datetime(oos_from)
        df_analysis['is_oos'] = df_analysis.index >= oos_timestamp
        pivot_group_cols = ['is_oos'] + active_binned_signal_cols
    
    return _process_bins_data(df_analysis, pivot_group_cols, forward_ret_col, pivot_aggfunc, 
                             None, show_stats, show_pnl_plot)

def _process_bins_data(df_analysis, pivot_group_cols, forward_ret_col, pivot_aggfunc, 
                      title_prefix=None, show_stats=True, show_pnl_plot=True):
    """Helper function to process binned data and generate stats and plots."""
    # Ensure forward_ret_col exists
    if forward_ret_col not in df_analysis.columns:
        raise ValueError(f"Forward return column '{forward_ret_col}' not found in DataFrame.")

    # Drop rows where any of the pivot grouping columns or the value column is NaN,
    # as pivot_table might handle them in ways that are not ideal for subsequent calculations
    cols_for_pivot_check = pivot_group_cols + [forward_ret_col]
    df_pivot_ready = df_analysis.dropna(subset=cols_for_pivot_check)

    if df_pivot_ready.empty:
        print(f"DataFrame is empty after dropping NaNs in {cols_for_pivot_check}. Cannot create pivot table.")
        return {"stats_df": pd.DataFrame(), "pnl_fig": None}

    try:
        # The pivot table will group by date (index) and then unstack the signal bins
        # Values are the forward returns for those specific date/bin combinations.
        # If multiple entries fall into the same date/bin, aggfunc handles it.
        returns_by_bin = pd.pivot_table(
            df_pivot_ready,
            index=df_pivot_ready.index.name or 'index', # Use index name or 'index' if None
            columns=pivot_group_cols,
            values=forward_ret_col,
            aggfunc=pivot_aggfunc
        )
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        print(f"Pivot settings: index='{df_pivot_ready.index.name or 'index'}', columns={pivot_group_cols}, values='{forward_ret_col}', aggfunc='{pivot_aggfunc}'")
        print("Sample of df_pivot_ready.head():")
        print(df_pivot_ready[cols_for_pivot_check + ([df_pivot_ready.index.name] if df_pivot_ready.index.name else [])].head())
        return {"stats_df": pd.DataFrame(), "pnl_fig": None}

    stats_df = returns_by_bin.apply(_calculate_group_stats).T # Transpose for stats per bin

    pnl_fig = None
    if show_pnl_plot:
        if not returns_by_bin.empty:
            cumulative_pnl = returns_by_bin.fillna(0).cumsum()
            
            # Prepare DataFrame for Plotly Express: flatten MultiIndex columns if they exist
            cumulative_pnl_for_plot = cumulative_pnl.copy()
            if isinstance(cumulative_pnl_for_plot.columns, pd.MultiIndex):
                # Convert tuple column names to string, e.g., (False, 0, 0) -> "F_0_0"
                # A short representation for boolean False/True could be F/T or 0/1 if preferred
                # Using str() for each element and joining with '_' is robust.
                cumulative_pnl_for_plot.columns = [
                    '_'.join(map(str, col_tuple)).strip()
                    for col_tuple in cumulative_pnl_for_plot.columns.values
                ]

            title = f"Cumulative P&L by Bins of {', '.join([col[:-4] for col in pivot_group_cols if col.endswith('_bin')])}"
            if title_prefix:
                title = f"{title_prefix} {title}"
                
            pnl_fig = px.line(cumulative_pnl_for_plot, title=title)
            if _is_notebook():
                pnl_fig.show()
            else:
                print(f"\nCumulative P&L Plot ({title_prefix or 'combined'} data not displayed in non-notebook environment):")
                print("Figure object returned in results['pnl_fig']")

        else:
            print("Cannot generate P&L plot: Pivoted returns are empty.")

    if show_stats:
        if not stats_df.empty:
            title = f"Statistics for {forward_ret_col} by Bins of {', '.join([col[:-4] for col in pivot_group_cols if col.endswith('_bin')])}"
            if title_prefix:
                title = f"{title_prefix} {title}"
                
            if _is_notebook():
                display(HTML(f"<h3>{title}</h3>"))
                display(stats_df.style.format("{:.3f}").background_gradient(
                #cmap='viridis'
                ))
            else:
                print(f"\n{title}:")
                print(tabulate(stats_df.round(3), headers='keys', tablefmt='psql'))
        else:
            print("Cannot display stats: Statistics DataFrame is empty.")
            
    return {"stats_df": stats_df, "pnl_fig": pnl_fig}

# Example of how to make it available in oequant.research.signal
# from .quantile_analysis import research_signal_bins
# would go into oequant/research/signal/__init__.py 