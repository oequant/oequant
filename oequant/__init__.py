from oequant.data import get_data
from oequant.backtesting import backtest, BacktestResult
from oequant.evaluations import calculate_statistics
from oequant.charting import plot_results

__all__ = ['get_data', 'backtest', 'BacktestResult', 'calculate_statistics', 'plot_results']
__version__ = "0.1.0" # From pyproject.toml 