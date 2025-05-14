import logging
import numpy as np
np.NaN = np.nan

import pandas_ta as ta

# Configure basic logging for the package
# Users can override this by configuring logging themselves before importing oequant
logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize pandas extensions
try:
    from .utils import pandas_extensions
    pandas_extensions.init_pandas_extensions()
except ImportError as e:
    # Log a warning if pandas_extensions or its dependencies are missing, 
    # but allow oequant to be imported without them if core functionality doesn't strictly require it.
    # This is a design choice: fail silently for extensions or fail loudly.
    # For now, let's warn, as extensions might be optional for some users.
    logging.getLogger(__name__).warning(
        f"Failed to initialize pandas extensions. Some plotting or utility features might not be available. Error: {e}"
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
    "__version__",
    "ta"
] 