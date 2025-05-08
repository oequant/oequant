import logging

# Configure basic logging for the package
# Users can override this by configuring logging themselves before importing oequant
logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set a more specific default level for oequant loggers if desired, 
# though basicConfig already sets the root logger level.
# logging.getLogger(__name__).setLevel(logging.WARNING) # Or getLogger('oequant')

# --- Public API --- 
from .data.core import get_data
from .backtesting.core import backtest
from .backtesting.results import BacktestResult
from .evaluations.core import calculate_statistics
from .charting.core import plot_results, plot_results_2

__version__ = "0.1.0" # Example version

__all__ = [
    "get_data",
    "backtest",
    "BacktestResult",
    "calculate_statistics",
    "plot_results",
    "plot_results_2",
    "__version__"
] 