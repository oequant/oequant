import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import show
from bokeh.layouts import LayoutDOM

# Removed top-level imports causing circular dependency
# from oequant.evaluations.core import calculate_statistics
# from oequant.charting.core import plot_results 

class BacktestResult:
    """
    Stores the results of a backtest.

    Attributes:
        trades (pd.DataFrame): DataFrame containing details of each trade.
        returns (pd.DataFrame): DataFrame containing bar-by-bar mark-to-market returns and positions.
        initial_capital (float): Initial capital for the backtest.
        final_equity (float): Equity at the end of the backtest.
        ohlcv_data (pd.DataFrame): The original OHLCV data used for the backtest.
    """
    def __init__(self, trades: pd.DataFrame, returns: pd.DataFrame, initial_capital: float, final_equity: float, ohlcv_data: pd.DataFrame):
        self.trades = trades
        self.returns = returns
        self.initial_capital = initial_capital
        self.final_equity = final_equity
        self.ohlcv_data = ohlcv_data # Store original data
        self._stats = None # Cache for statistics

    def statistics(self, PnL_type: str = 'net', risk_free_rate_annual: float = 0.0) -> dict:
        """
        Calculates and returns performance statistics for the backtest.

        Args:
            PnL_type (str, optional): Use 'gross' or 'net' P&L. Defaults to 'net'.
            risk_free_rate_annual (float, optional): Annual risk-free rate. Defaults to 0.0.

        Returns:
            dict: Dictionary of performance statistics.
        """
        # Import moved inside method
        from oequant.evaluations.core import calculate_statistics
        
        args_key = (PnL_type, risk_free_rate_annual)
        if self._stats is None or self._stats.get('_args_key') != args_key:
            self._stats = calculate_statistics(self, PnL_type=PnL_type, risk_free_rate_annual=risk_free_rate_annual)
            self._stats['_args_key'] = args_key # Store args used for caching
        
        # Return a copy without the internal args key
        stats_copy = self._stats.copy()
        stats_copy.pop('_args_key', None)
        return stats_copy

    def plot(
        self,
        price_col: str = 'close', 
        indicators_price: list = None, 
        indicators_other: list = None,
        show_ohlc: bool = False,
        plot_width: int = 1000
    ) -> LayoutDOM:
        """
        Generates a plot of the backtest results using Bokeh.

        Args:
            price_col (str): Column in ohlcv_data for main price plot (default 'close').
            indicators_price (list, optional): Columns from ohlcv_data for price chart overlay.
            indicators_other (list, optional): Columns from ohlcv_data for separate subplots.
            show_ohlc (bool, optional): If True, attempts to plot OHLC data. Defaults to False.
            plot_width (int, optional): Width of the plot.

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
            plot_width=plot_width
        )
        
    def report(self, show_plot=True, stats_args=None, plot_args=None):
        """
        Generates a standard report containing statistics and optionally a plot.

        Args:
            show_plot (bool, optional): Whether to generate and display the plot. Defaults to True.
            stats_args (dict, optional): Arguments to pass to the .statistics() method.
            plot_args (dict, optional): Arguments to pass to the .plot() method.
        """
        # Import moved inside method (needed for plt.show)
        from bokeh.plotting import show
        
        if stats_args is None: stats_args = {}
        if plot_args is None: plot_args = {}
        
        # Calculate statistics
        stats_dict = self.statistics(**stats_args)
        
        # Print statistics
        print("--- Backtest Report ---")
        print(f"Initial Capital: {self.initial_capital:,.2f}")
        print(f"Final Equity:    {self.final_equity:,.2f}")
        print("\n--- Performance Metrics --- ('{stats_args.get('PnL_type', 'net')}' PnL)")
        for key, value in stats_dict.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, float):
                if 'pct' in key:
                    print(f"{label:<25}: {value:,.2f}%")
                elif np.isinf(value) or np.isnan(value):
                     print(f"{label:<25}: {value}")
                else:
                    print(f"{label:<25}: {value:,.4f}")
            else:
                 print(f"{label:<25}: {value}")
        print("-----------------------")

        # Generate and show plot
        if show_plot:
            fig = self.plot(**plot_args)
            show(fig)
        else:
            fig = None

        # Return both for programmatic access if needed
        return stats_dict, fig 

    def __repr__(self):
        # Keep repr simple, encourage use of .statistics() or .report()
        return (f"BacktestResult(Initial Capital={self.initial_capital:,.2f}, "
                f"Final Equity={self.final_equity:,.2f}, Total Trades={len(self.trades)})") 