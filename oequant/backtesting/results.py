import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import show
from bokeh.layouts import LayoutDOM
from typing import Optional, Tuple # Added Tuple
from tabulate import tabulate # Added import

# Removed top-level imports causing circular dependency
# from oequant.evaluations.core import calculate_statistics
# from oequant.charting.core import plot_results 

# Forward declaration for type hinting self-reference
SelfBacktestResult = Optional['BacktestResult']
# Type hint for the cache tuple
CacheTuple = Optional[Tuple[tuple, pd.Series]]

# --- Helper for pretty stat names ---
_PRETTY_STAT_NAMES = {
    'cagr_pct': 'CAGR (%)',
    'return_per_trade_pct': 'Avg Trade (%)',
    'sharpe_ratio': 'Sharpe Ratio',
    'sortino_ratio': 'Sortino Ratio',
    'serenity_ratio': 'Serenity Ratio',
    'pct_in_position': 'Time in Market (%)',
    'max_dd_pct': 'Max Drawdown (%)',
    'cagr_to_max_dd': 'CAGR / Max DD',
    'total_trades': 'Total Trades',
    'win_rate_pct': 'Win Rate (%)',
    'loss_rate_pct': 'Loss Rate (%)',
    'avg_win_trade_pct': 'Avg Win Trade (%)',
    'avg_loss_trade_pct': 'Avg Loss Trade (%)',
    'profit_factor': 'Profit Factor',
    'avg_bars_held': 'Avg Bars Held'
}
_PERCENTAGE_KEYS = {k for k, v in _PRETTY_STAT_NAMES.items() if '%' in v}

def _prettify_stat_name(key):
    return _PRETTY_STAT_NAMES.get(key, key.replace('_', ' ').title())
# --- End Helper ---

class BacktestResult:
    """
    Stores the results of a backtest.

    Attributes:
        trades (pd.DataFrame): DataFrame containing details of each trade.
        returns (pd.DataFrame): DataFrame containing bar-by-bar mark-to-market returns and positions.
        initial_capital (float): Initial capital for the backtest.
        final_equity (float): Equity at the end of the backtest.
        ohlcv_data (pd.DataFrame): The original OHLCV data used for the backtest.
        benchmark_res (Optional[BacktestResult]): Stores the BacktestResult for a benchmark run, if any.
    """
    def __init__(self, trades: pd.DataFrame, returns: pd.DataFrame, initial_capital: float, final_equity: float, ohlcv_data: pd.DataFrame, benchmark_res: SelfBacktestResult = None):
        self.trades = trades
        self.returns = returns
        self.initial_capital = initial_capital
        self.final_equity = final_equity
        self.ohlcv_data = ohlcv_data # Store original data
        self._cached_stats_tuple: CacheTuple = None # Cache for statistics (key, series)
        self.benchmark_res = benchmark_res # Store benchmark result

    def statistics(self, PnL_type: str = 'net', risk_free_rate_annual: float = 0.0) -> pd.Series:
        """
        Calculates and returns performance statistics for the backtest.
        Uses internal caching.

        Args:
            PnL_type (str, optional): Use 'gross' or 'net' P&L. Defaults to 'net'.
            risk_free_rate_annual (float, optional): Annual risk-free rate. Defaults to 0.0.

        Returns:
            pd.Series: Series of performance statistics.
        """
        from oequant.evaluations.core import calculate_statistics
        
        args_key = (PnL_type, risk_free_rate_annual)
        if self.benchmark_res:
            args_key += (self.benchmark_res.final_equity,) # Include for cache invalidation

        # Check cache
        if self._cached_stats_tuple is not None and self._cached_stats_tuple[0] == args_key:
            return self._cached_stats_tuple[1].copy() # Return copy of cached Series

        # Cache miss or invalid: recalculate
        new_stats_series = calculate_statistics(self, PnL_type=PnL_type, risk_free_rate_annual=risk_free_rate_annual)
        
        # Update cache
        self._cached_stats_tuple = (args_key, new_stats_series)
        
        return new_stats_series.copy() # Return copy of newly calculated Series

    def plot(
        self,
        price_col: str = 'close', 
        indicators_price: list = None, 
        indicators_other: list = None,
        show_ohlc: bool = False,
        plot_width: int = 1000,
        show_benchmark: bool = True,
        main_price_plot_height: int = 400,
        per_indicator_plot_height: int = 80,
        plot_theme: str = "dark"
    ) -> LayoutDOM:
        """
        Generates a plot of the backtest results using Bokeh.

        Args:
            price_col (str): Column in ohlcv_data for main price plot (default 'close').
            indicators_price (list, optional): Columns from ohlcv_data for price chart overlay.
            indicators_other (list, optional): Columns from ohlcv_data for separate subplots.
            show_ohlc (bool, optional): If True, attempts to plot OHLC data. Defaults to False.
            plot_width (int, optional): Width of the plot.
            show_benchmark (bool, optional): If True and benchmark available, plot benchmark equity. Defaults to True.
            main_price_plot_height (int, optional): Height of the main price plot. Defaults to 400.
            per_indicator_plot_height (int, optional): Height of each secondary indicator plot. Defaults to 80.
            plot_theme (str, optional): Theme for the plot ("dark" or "light"). Defaults to "dark".

        Returns:
            bokeh.layouts.LayoutDOM: The Bokeh layout object.
        """
        # Import moved inside method
        from oequant.charting.core import plot_results
        
        return plot_results(
            result=self,
            price_col=price_col,
            indicators_price=indicators_price,
            indicators_other=indicators_other,
            show_ohlc=show_ohlc,
            plot_width=plot_width,
            show_benchmark=show_benchmark,
            main_price_plot_height=main_price_plot_height,
            per_indicator_plot_height=per_indicator_plot_height,
            plot_theme=plot_theme
        )
        
    def report(self, show_plot=True, stats_args=None, plot_args=None, show_benchmark_in_report: bool = True, table_format: str = 'pipe'):
        """
        Generates a standard report containing statistics and optionally a plot.

        Args:
            show_plot (bool, optional): Whether to generate and display the plot. Defaults to True.
            stats_args (dict, optional): Arguments to pass to the .statistics() method.
            plot_args (dict, optional): Arguments to pass to the .plot() method.
            show_benchmark_in_report (bool, optional): Whether to include benchmark stats in the table. Defaults to True.
            table_format (str, optional): Format string for tabulate (e.g., 'grid', 'simple', 'pipe'). Defaults to 'pipe'.
        """
        from bokeh.plotting import show # Import moved inside method
        
        if stats_args is None: stats_args = {}
        if plot_args is None: plot_args = {}
        
        # --- Get Dates --- 
        start_date_str = "N/A"
        end_date_str = "N/A"
        duration_str = "N/A"
        if not self.returns.empty and isinstance(self.returns.index, pd.DatetimeIndex):
            start_date = self.returns.index[0]
            end_date = self.returns.index[-1]
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            duration = end_date - start_date
            duration_str = str(duration)
        
        # --- Statistics Table Generation ---
        strat_stats: pd.Series = self.statistics(**stats_args)
        bench_stats: Optional[pd.Series] = None
        
        stats_df_dict = {'Strategy': strat_stats}
        if self.benchmark_res and show_benchmark_in_report:
            bench_stats = self.benchmark_res.statistics(**stats_args)
            stats_df_dict['Benchmark'] = bench_stats
            
        stats_df = pd.DataFrame(stats_df_dict)
        
        # Prettify index
        stats_df.index = stats_df.index.map(_prettify_stat_name)
        
        # Store original keys before prettifying index for formatting check
        original_keys = strat_stats.index if strat_stats is not None else pd.Index([])
        key_map = {v: k for k, v in _PRETTY_STAT_NAMES.items()} # Map pretty back to original

        # Format numerical columns
        for col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(
                lambda x: f"{x:,.2f}%" if isinstance(x, (int, float))
                          # Check original key to see if it should be % formatted
                          and key_map.get(stats_df[stats_df[col] == x].index[0] if not pd.isna(x) else '-', '-') in _PERCENTAGE_KEYS
                          # Fallback formatting for floats (non-pct, non-inf/nan)
                          else (f"{x:,.4f}" if isinstance(x, (int, float)) and not (np.isnan(x) or np.isinf(x))
                                # Fallback for non-float or inf/nan
                                else str(x) if not pd.isna(x) else 'nan') 
            )
            
        # Print Header Info
        print("--- Backtest Report ---")
        print(f"Start Date:             {start_date_str}")
        print(f"End Date:               {end_date_str}")
        print(f"Duration:               {duration_str}")
        print(f"Initial Capital:        {self.initial_capital:,.2f}")
        print(f"Final Equity:           {self.final_equity:,.2f}")
        if bench_stats is not None:
            print(f"Benchmark Final Equity: {self.benchmark_res.final_equity:,.2f}")
        print("\n--- Performance Metrics ---")
        
        # Print the table using tabulate
        print(tabulate(stats_df, headers='keys', tablefmt=table_format, stralign="right"))
        # --- End Statistics Table Generation ---

        # Generate and show plot
        if show_plot:
            fig = self.plot(**plot_args)
            show(fig)
        else:
            fig = None

        # Return both for programmatic access if needed
        return strat_stats, fig 

    def __repr__(self):
        stats = self.statistics() # Ensure stats are calculated if needed by representation
        repr_str = (
            f"<BacktestResult: Equity {self.initial_capital:,.2f} -> {self.final_equity:,.2f}, "
            f"Trades: {len(self.trades)}, "
            f"CAGR: {stats.get('cagr_pct', 'N/A') if isinstance(stats.get('cagr_pct'), (int, float)) else 'N/A'}, " # defensive formatting
            f"Max DD: {stats.get('max_dd_pct', 'N/A') if isinstance(stats.get('max_dd_pct'), (int, float)) else 'N/A'}, "
            f"Sharpe: {stats.get('sharpe_ratio', 'N/A') if isinstance(stats.get('sharpe_ratio'), (int, float)) else 'N/A'}, "
            f"Time in Pos: {stats.get('pct_in_position', 'N/A') if isinstance(stats.get('pct_in_position'), (int, float)) else 'N/A'}%"
        )
        if self.benchmark_res:
            # Ensure benchmark stats are also defensively formatted
            bench_stats = self.benchmark_res.statistics() # Calculate benchmark stats for its representation
            bench_cagr = bench_stats.get('cagr_pct', 'N/A')
            bench_cagr_str = f"{bench_cagr:.2f}%" if isinstance(bench_cagr, (int, float)) else str(bench_cagr)
            repr_str += f" | Benchmark CAGR: {bench_cagr_str}"
        repr_str += ">"
        return repr_str

    def _repr_html_(self):
        stats = self.statistics()
        html = "<h4>BacktestResult</h4>"
        html += "<table style='border-collapse: collapse; border: 1px solid black;'>"
        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Initial Capital</td><td style='border: 1px solid black; padding: 5px;'>{self.initial_capital:,.2f}</td></tr>"
        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Final Equity</td><td style='border: 1px solid black; padding: 5px;'>{self.final_equity:,.2f}</td></tr>"
        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Total Trades</td><td style='border: 1px solid black; padding: 5px;'>{len(self.trades)}</td></tr>"
        
        def format_stat(value, is_pct=False):
            if isinstance(value, (int, float)) and not (np.isinf(value) or np.isnan(value)):
                return f"{value:.2f}%" if is_pct else f"{value:.2f}"
            return str(value)

        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>CAGR</td><td style='border: 1px solid black; padding: 5px;'>{format_stat(stats.get('cagr_pct'), is_pct=True)}</td></tr>"
        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Max Drawdown</td><td style='border: 1px solid black; padding: 5px;'>{format_stat(stats.get('max_dd_pct'), is_pct=True)}</td></tr>"
        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Sharpe Ratio</td><td style='border: 1px solid black; padding: 5px;'>{format_stat(stats.get('sharpe_ratio'))}</td></tr>"
        html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Time in Position</td><td style='border: 1px solid black; padding: 5px;'>{format_stat(stats.get('pct_in_position'), is_pct=True)}</td></tr>"
        
        if self.benchmark_res:
            bench_stats = self.benchmark_res.statistics()
            html += "<tr><td colspan='2' style='border: 1px solid black; padding: 5px; text-align: center; background-color: #f0f0f0;'><b>Benchmark</b></td></tr>"
            html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Benchmark Initial Capital</td><td style='border: 1px solid black; padding: 5px;'>{self.benchmark_res.initial_capital:,.2f}</td></tr>"
            html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Benchmark Final Equity</td><td style='border: 1px solid black; padding: 5px;'>{self.benchmark_res.final_equity:,.2f}</td></tr>"
            html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Benchmark CAGR</td><td style='border: 1px solid black; padding: 5px;'>{format_stat(bench_stats.get('cagr_pct'), is_pct=True)}</td></tr>"
            html += f"<tr><td style='border: 1px solid black; padding: 5px;'>Benchmark Max Drawdown</td><td style='border: 1px solid black; padding: 5px;'>{format_stat(bench_stats.get('max_dd_pct'), is_pct=True)}</td></tr>"

        html += "</table>"
        return html 