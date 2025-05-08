import pandas as pd
import numpy as np
import pytest
# import matplotlib.pyplot as plt # No longer needed
from bokeh.layouts import LayoutDOM # Import Bokeh layout type

from oequant.backtesting.results import BacktestResult
from oequant.charting import plot_results

# Re-use or create fixtures similar to evaluations tests
@pytest.fixture
def sample_backtest_result_for_plotting():
    num_days = 20
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='B')
    
    # Original OHLCV data
    ohlc_data = pd.DataFrame(index=dates)
    ohlc_data['open'] = np.linspace(100, 110, num_days) + np.random.randn(num_days) * 0.5
    ohlc_data['high'] = ohlc_data['open'] + np.random.rand(num_days) * 2
    ohlc_data['low'] = ohlc_data['open'] - np.random.rand(num_days) * 2
    ohlc_data['close'] = ohlc_data['open'] + (np.random.rand(num_days) - 0.5) * 2
    ohlc_data['volume'] = np.random.randint(1000, 5000, num_days)
    ohlc_data['sma_10'] = ohlc_data['close'].rolling(10).mean()
    ohlc_data['rsi_14'] = 100 - (100 / (1 + np.random.rand(num_days)*10)) # Dummy RSI

    # Mock returns data
    returns_data = {
        'equity': np.linspace(10000, 10500, num_days),
        'position_unit': ([0]*5 + [10]*10 + [0]*5),
        'position_usd': ([0]*5 + [10 * ohlc_data['close'].iloc[i] for i in range(5, 15)] + [0]*5),
        'return_net_frac': np.random.randn(num_days) * 0.001
    }
    returns_df = pd.DataFrame(returns_data, index=dates)
    for col in ['return_gross_frac', 'return_gross_currency', 'return_net_currency']:
        returns_df[col] = 0.0 # Fill dummy

    # Mock trades data
    trades_data = [{
        'trade_number': 1, 'entry_time': dates[5], 'entry_price': ohlc_data['close'].iloc[5],
        'quantity': 10, 'exit_time': dates[15], 'exit_price': ohlc_data['close'].iloc[15],
        'pnl_net_currency': 50.0 # dummy
    }]
    trades_df = pd.DataFrame(trades_data)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    return BacktestResult(trades=trades_df, returns=returns_df, initial_capital=10000.0, 
                            final_equity=returns_df['equity'].iloc[-1], ohlcv_data=ohlc_data)

class TestCharting:
    def test_plot_results_runs(self, sample_backtest_result_for_plotting):
        """Test that plotting runs without errors for basic case."""
        result = sample_backtest_result_for_plotting
        try:
            layout = plot_results(result)
            assert isinstance(layout, LayoutDOM)
            # Basic plot has Price + Equity = 2 children in the gridplot
            assert len(layout.children) == 2 
        except Exception as e:
            pytest.fail(f"plot_results failed with basic input: {e}")

    def test_plot_results_with_indicators(self, sample_backtest_result_for_plotting):
        """Test plotting with price and other indicators."""
        result = sample_backtest_result_for_plotting
        try:
            layout = plot_results(result, 
                               indicators_price=['sma_10'], 
                               indicators_other=['rsi_14', 'volume'])
            assert isinstance(layout, LayoutDOM)
            # Check number of figures (Price + Equity + 2 others = 4)
            assert len(layout.children) == 4 
        except Exception as e:
            pytest.fail(f"plot_results failed with indicators: {e}")

    def test_plot_results_ohlc(self, sample_backtest_result_for_plotting):
        """Test plotting with OHLC enabled."""
        result = sample_backtest_result_for_plotting
        try:
            layout = plot_results(result, show_ohlc=True)
            assert isinstance(layout, LayoutDOM)
            assert len(layout.children) == 2 # Price + Equity
        except Exception as e:
            pytest.fail(f"plot_results failed with show_ohlc=True: {e}")

    def test_plot_results_no_trades(self):
        """Test plotting when there are no trades."""
        num_days = 10
        dates = pd.date_range(start='2023-01-01', periods=num_days, freq='B')
        ohlc_data = pd.DataFrame({'close': np.linspace(100, 105, num_days)}, index=dates)
        returns_df = pd.DataFrame({'equity': [10000]*num_days}, index=dates)
        for col in ['position_unit', 'position_usd', 'return_gross_frac', 'return_net_frac', 'return_gross_currency', 'return_net_currency']:
            returns_df[col] = 0.0
        trades_df = pd.DataFrame()
        result = BacktestResult(trades=trades_df, returns=returns_df, initial_capital=10000, final_equity=10000, ohlcv_data=ohlc_data)
        try:
            layout = plot_results(result)
            assert isinstance(layout, LayoutDOM)
            assert len(layout.children) == 2 # Price + Equity
        except Exception as e:
            pytest.fail(f"plot_results failed with no trades: {e}") 