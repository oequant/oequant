import pandas as pd
import numpy as np
import pytest
# import matplotlib.pyplot as plt # No longer needed
from bokeh.layouts import LayoutDOM, column # Import Bokeh layout type and a simple layout
from bokeh.plotting import figure # Import figure to create a simple layout
from unittest.mock import patch, MagicMock

from oequant.backtesting.results import BacktestResult
# Need data creation helpers or fixtures

@pytest.fixture
def sample_backtest_result_for_methods():
    # Use a fixture similar to the one in test_charting or test_evaluations
    num_days = 20
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='B')
    
    ohlc_data = pd.DataFrame(index=dates)
    ohlc_data['open'] = np.linspace(100, 110, num_days) + np.random.randn(num_days) * 0.5
    ohlc_data['close'] = ohlc_data['open'] + (np.random.rand(num_days) - 0.5) * 2
    
    returns_data = {
        'equity': np.linspace(10000, 10500, num_days),
        'position_unit': ([0]*5 + [10]*10 + [0]*5),
        'return_net_frac': np.random.randn(num_days) * 0.001
    }
    returns_df = pd.DataFrame(returns_data, index=dates)
    # Fill necessary columns for stats calculation
    returns_df['position_usd'] = returns_df['position_unit'] * ohlc_data['close']
    returns_df['return_gross_frac'] = returns_df['return_net_frac'] # Simplification
    returns_df['return_gross_currency'] = 0.0
    returns_df['return_net_currency'] = 0.0

    trades_data = [{
        'trade_number': 1, 'entry_time': dates[5], 'entry_price': ohlc_data['close'].iloc[5],
        'quantity': 10, 'exit_time': dates[15], 'exit_price': ohlc_data['close'].iloc[15],
        'pnl_net_currency': 50.0, 'pnl_net_frac': 0.005 # dummy
    }]
    trades_df = pd.DataFrame(trades_data)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['pnl_gross_currency'] = trades_df['pnl_net_currency'] # Simplification
    trades_df['pnl_gross_frac'] = trades_df['pnl_net_frac'] # Simplification

    return BacktestResult(trades=trades_df, returns=returns_df, initial_capital=10000.0, 
                            final_equity=returns_df['equity'].iloc[-1], ohlcv_data=ohlc_data)

class TestBacktestResultMethods:

    def test_statistics_method(self, sample_backtest_result_for_methods):
        """Test the .statistics() method calls calculate_statistics."""
        result = sample_backtest_result_for_methods
        # Patch the function in its original module
        with patch('oequant.evaluations.core.calculate_statistics') as mock_calc_stats:
            mock_calc_stats.return_value = {'test_stat': 123, '_args_key': ('net', 0.0)} # Keep _args_key for internal logic test
            stats = result.statistics(PnL_type='net', risk_free_rate_annual=0.0)
            mock_calc_stats.assert_called_once_with(result, PnL_type='net', risk_free_rate_annual=0.0)
            assert stats == {'test_stat': 123}
            
            # Test caching - call again, should not call calculate_statistics again
            mock_calc_stats.reset_mock()
            stats_cached = result.statistics(PnL_type='net', risk_free_rate_annual=0.0)
            mock_calc_stats.assert_not_called()
            assert stats_cached == {'test_stat': 123}

            # Test cache invalidation - call with different args
            mock_calc_stats.reset_mock()
            mock_calc_stats.return_value = {'test_stat_gross': 456, '_args_key': ('gross', 0.0)}
            stats_gross = result.statistics(PnL_type='gross', risk_free_rate_annual=0.0)
            mock_calc_stats.assert_called_once_with(result, PnL_type='gross', risk_free_rate_annual=0.0)
            assert stats_gross == {'test_stat_gross': 456}

    def test_plot_method(self, sample_backtest_result_for_methods):
        """Test the .plot() method calls plot_results."""
        result = sample_backtest_result_for_methods
        # Mock a Bokeh layout object
        mock_layout = MagicMock(spec=LayoutDOM)
        plot_args = {'price_col': 'close', 'show_ohlc': True}
        
        # Patch the function in its original module
        with patch('oequant.charting.core.plot_results') as mock_plot:
            mock_plot.return_value = mock_layout
            layout = result.plot(**plot_args)
            # Check against the full signature including defaults passed by the method
            expected_call_args = {
                'result': result,
                'price_col': 'close',
                'indicators_price': None, # Default from .plot()
                'indicators_other': None, # Default from .plot()
                'show_ohlc': True, # From plot_args
                # 'figsize': (15, 10)    # Removed figsize
                'plot_width': 1000 # Default from .plot()
            }
            mock_plot.assert_called_once_with(**expected_call_args)
            assert layout is mock_layout

    # Mock bokeh.plotting.show and print to test report
    @patch('bokeh.plotting.show') # Correct patch target
    @patch('builtins.print')
    def test_report_method(self, mock_print, mock_bokeh_show, sample_backtest_result_for_methods):
        """Test the .report() method calls statistics and plot."""
        result = sample_backtest_result_for_methods
        mock_stats_dict = {'cagr_pct': 10.5, 'total_trades': 1}
        # Use a real (empty) Bokeh layout instead of MagicMock for serialization
        mock_layout = column() # An empty column layout is serializable
        
        # Mock the instance methods .statistics() and .plot()
        with patch.object(result, 'statistics', return_value=mock_stats_dict) as mock_stats_method, \
             patch.object(result, 'plot', return_value=mock_layout) as mock_plot_method:
            
            # Test report with plot
            plot_args_test = {'show_ohlc': True}
            stats_out, layout_out = result.report(show_plot=True, stats_args={'PnL_type': 'gross'}, plot_args=plot_args_test)
            
            mock_stats_method.assert_called_once_with(PnL_type='gross')
            mock_plot_method.assert_called_once_with(**plot_args_test)
            mock_bokeh_show.assert_called_once_with(mock_layout) # Check that bokeh.show is called
            mock_print.assert_called() # Check that stats are printed
            assert stats_out is mock_stats_dict
            assert layout_out is mock_layout

            # Test report without plot
            mock_stats_method.reset_mock()
            mock_plot_method.reset_mock()
            mock_bokeh_show.reset_mock()
            stats_out_no_plot, layout_out_no_plot = result.report(show_plot=False)
            mock_stats_method.assert_called_once_with() # Default args
            mock_plot_method.assert_not_called()
            mock_bokeh_show.assert_not_called()
            assert layout_out_no_plot is None 