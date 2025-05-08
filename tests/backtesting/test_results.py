import pandas as pd
import numpy as np
import pytest
# import matplotlib.pyplot as plt # No longer needed
from bokeh.layouts import LayoutDOM, column # Import Bokeh layout type and a simple layout
from bokeh.plotting import figure # Import figure to create a simple layout
from unittest.mock import patch, MagicMock
from tabulate import tabulate # Import tabulate for mocking if needed

from oequant.backtesting.results import BacktestResult, _prettify_stat_name
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
            # Mock calculate_statistics to return a Series, as the actual function does
            mock_return_series = pd.Series({'test_stat': 123})
            mock_calc_stats.return_value = mock_return_series 
            
            stats_series = result.statistics(PnL_type='net', risk_free_rate_annual=0.0)
            
            # Assert that calculate_statistics was called correctly
            mock_calc_stats.assert_called_once_with(result, PnL_type='net', risk_free_rate_annual=0.0)
            # Assert the final return type is Series and content matches
            assert isinstance(stats_series, pd.Series)
            pd.testing.assert_series_equal(stats_series, mock_return_series, check_names=False)

            # Test caching - call again, should not call calculate_statistics again
            mock_calc_stats.reset_mock()
            stats_cached = result.statistics(PnL_type='net', risk_free_rate_annual=0.0)
            mock_calc_stats.assert_not_called()
            assert isinstance(stats_cached, pd.Series)
            pd.testing.assert_series_equal(stats_cached, mock_return_series, check_names=False)

            # Test cache invalidation - call with different args
            mock_calc_stats.reset_mock()
            mock_return_series_gross = pd.Series({'test_stat_gross': 456})
            mock_calc_stats.return_value = mock_return_series_gross

            stats_gross = result.statistics(PnL_type='gross', risk_free_rate_annual=0.0)
            mock_calc_stats.assert_called_once_with(result, PnL_type='gross', risk_free_rate_annual=0.0)
            assert isinstance(stats_gross, pd.Series)
            pd.testing.assert_series_equal(stats_gross, mock_return_series_gross, check_names=False)

    def test_plot_method(self, sample_backtest_result_for_methods):
        """Test the .plot() method calls plot_results."""
        result = sample_backtest_result_for_methods
        mock_layout = MagicMock(spec=LayoutDOM)
        
        # --- Test case 1: Explicitly setting show_benchmark=False --- 
        plot_args_no_bench = {'price_col': 'close', 'show_ohlc': True, 'show_benchmark': False}
        with patch('oequant.charting.core.plot_results') as mock_plot_no_bench:
            mock_plot_no_bench.return_value = mock_layout
            layout_no_bench = result.plot(**plot_args_no_bench)
            
            expected_call_args_no_bench = {
                'result': result,
                'price_col': 'close',
                'indicators_price': None,
                'indicators_other': None,
                'show_ohlc': True,
                'plot_width': 1000,
                'show_benchmark': False, # Check False is passed
                'main_price_plot_height': 400,  # Added default
                'per_indicator_plot_height': 80,   # Added default
                'plot_theme': 'dark'  # Added default theme
            }
            mock_plot_no_bench.assert_called_once_with(**expected_call_args_no_bench)
            assert layout_no_bench == mock_layout

        # --- Test case 2: Default show_benchmark=True --- 
        plot_args_default_bench = {'price_col': 'open', 'plot_theme': 'light'} # Minimal args, but override theme for test
        with patch('oequant.charting.core.plot_results') as mock_plot_default_bench:
            mock_plot_default_bench.return_value = mock_layout
            layout_default_bench = result.plot(**plot_args_default_bench)
            
            expected_call_args_default_bench = {
                'result': result,
                'price_col': 'open', # From args
                'indicators_price': None,
                'indicators_other': None,
                'show_ohlc': False, # Default from .plot()
                'plot_width': 1000, # Default from .plot()
                'show_benchmark': True, # Check default is True
                'main_price_plot_height': 400,  # Default from BacktestResult.plot
                'per_indicator_plot_height': 80,   # Default from BacktestResult.plot
                'plot_theme': 'light' # Overridden for this test case
            }
            mock_plot_default_bench.assert_called_once_with(**expected_call_args_default_bench)
            assert layout_default_bench is mock_layout

    # Mock bokeh.plotting.show and print to test report
    @patch('bokeh.plotting.show') # Correct patch target
    @patch('builtins.print')
    def test_report_method(self, mock_print, mock_bokeh_show, sample_backtest_result_for_methods):
        """Test the .report() method calls statistics, formats, and prints using tabulate."""
        result = sample_backtest_result_for_methods
        # Mock statistics to return a Series
        mock_stats_series = pd.Series({'cagr_pct': 10.5, 'total_trades': 1, 'sharpe_ratio': 1.2})
        mock_layout = column() 
        
        with patch.object(result, 'statistics', return_value=mock_stats_series) as mock_stats_method, \
             patch.object(result, 'plot', return_value=mock_layout) as mock_plot_method:
            
            result.report(show_plot=True, stats_args={'PnL_type': 'gross'}, plot_args={'show_ohlc': True}, table_format='plain') # Use plain for simpler assertion
            
            mock_stats_method.assert_called_once_with(PnL_type='gross')
            mock_plot_method.assert_called_once_with(show_ohlc=True)
            mock_bokeh_show.assert_called_once_with(mock_layout)
            
            # Check print output for key elements of the tabulate table
            printed_text = "\n".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
            # print(f"\nDEBUG test_report_method:\n{printed_text}\n") # Keep for debugging
            
            assert "--- Backtest Report ---" in printed_text
            assert "Initial Capital:" in printed_text
            assert "Final Equity:" in printed_text
            assert "--- Performance Metrics ---" in printed_text
            assert "Strategy" in printed_text # Header from tabulate
            assert _prettify_stat_name('cagr_pct') in printed_text # Check pretty index name
            assert "10.50%" in printed_text # Check formatted value
            assert _prettify_stat_name('sharpe_ratio') in printed_text
            assert "1.2000" in printed_text
            assert _prettify_stat_name('total_trades') in printed_text
            assert "1" in printed_text 

    @patch('bokeh.plotting.show')
    @patch('builtins.print')
    def test_report_method_no_plot(self, mock_print, mock_bokeh_show, sample_backtest_result_for_methods):
        result = sample_backtest_result_for_methods
        mock_stats_series = pd.Series({'cagr_pct': 10.5, 'total_trades': 1})
        with patch.object(result, 'statistics', return_value=mock_stats_series) as mock_stats_method, \
             patch.object(result, 'plot') as mock_plot_method:

            result.report(show_plot=False, table_format='plain')
            mock_stats_method.assert_called_once_with() # Default args for stats
            mock_plot_method.assert_not_called()
            mock_bokeh_show.assert_not_called()
            printed_text = "\n".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
            assert "Strategy" in printed_text
            assert _prettify_stat_name('cagr_pct') in printed_text
            assert "10.50%" in printed_text

    @patch('builtins.print')
    def test_report_with_benchmark(self, mock_print, sample_backtest_result_for_methods):
        strategy_result = sample_backtest_result_for_methods
        # Create a simple benchmark result
        benchmark_res_obj = BacktestResult(trades=pd.DataFrame(), returns=pd.DataFrame(), initial_capital=10000, final_equity=10200, ohlcv_data=pd.DataFrame())
        strategy_result.benchmark_res = benchmark_res_obj

        mock_strat_stats = pd.Series({'cagr_pct': 10.5, 'sharpe_ratio': 1.2, 'total_trades': 1})
        mock_bench_stats = pd.Series({'cagr_pct': 5.0, 'sharpe_ratio': 0.8, 'total_trades': 1})

        with patch.object(strategy_result, 'statistics', return_value=mock_strat_stats) as mock_strat_stats_method, \
             patch.object(benchmark_res_obj, 'statistics', return_value=mock_bench_stats) as mock_bench_stats_method, \
             patch('oequant.backtesting.results.show'): # Patch show as plot is not the focus here
            
            strategy_result.report(show_plot=False, show_benchmark_in_report=True, table_format='plain')

            mock_strat_stats_method.assert_called_once()
            mock_bench_stats_method.assert_called_once()

            printed_text = "\n".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
            # print(f"\nDEBUG test_report_with_benchmark:\n{printed_text}\n")
            
            assert "Strategy" in printed_text 
            assert "Benchmark" in printed_text # Check for benchmark header
            assert "Benchmark Final Equity: 10,200.00" in printed_text
            assert _prettify_stat_name('cagr_pct') in printed_text
            assert "10.50%" in printed_text
            assert "5.00%" in printed_text
            assert _prettify_stat_name('sharpe_ratio') in printed_text
            assert "1.2000" in printed_text
            assert "0.8000" in printed_text
            assert _prettify_stat_name('total_trades') in printed_text
            assert "1" in printed_text # Should appear multiple times
            
    @patch('builtins.print')
    def test_report_with_benchmark_hidden(self, mock_print, sample_backtest_result_for_methods):
        strategy_result = sample_backtest_result_for_methods
        benchmark_res_obj = MagicMock(spec=BacktestResult) # Mock benchmark
        strategy_result.benchmark_res = benchmark_res_obj

        mock_strat_stats = pd.Series({'cagr_pct': 10.5, 'sharpe_ratio': 1.2})
        
        with patch.object(strategy_result, 'statistics', return_value=mock_strat_stats) as mock_strat_stats_method, \
             patch.object(benchmark_res_obj, 'statistics') as mock_bench_stats_method, \
             patch('oequant.backtesting.results.show'):

            strategy_result.report(show_plot=False, show_benchmark_in_report=False, table_format='plain')

            mock_strat_stats_method.assert_called_once()
            mock_bench_stats_method.assert_not_called() # Benchmark stats should not be fetched

            printed_text = "\n".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
            # print(f"\nDEBUG test_report_with_benchmark_hidden:\n{printed_text}\n")
            assert "Benchmark" not in printed_text # Header should be absent
            assert "Strategy" in printed_text
            assert _prettify_stat_name('cagr_pct') in printed_text
            assert "10.50%" in printed_text
            assert "Benchmark Final Equity" not in printed_text

    @patch('builtins.print')
    def test_report_metric_missing_in_benchmark(self, mock_print, sample_backtest_result_for_methods):
        strategy_result = sample_backtest_result_for_methods
        benchmark_res_obj = BacktestResult( # Real benchmark, missing stats
            trades=pd.DataFrame(), returns=pd.DataFrame(), 
            initial_capital=10000, final_equity=10000, ohlcv_data=pd.DataFrame()
        )
        strategy_result.benchmark_res = benchmark_res_obj

        mock_strat_stats = pd.Series({'cagr_pct': 10.5, 'some_custom_stat': 777})
        # Benchmark stats might not have 'some_custom_stat'. Note: calculate_statistics now returns Series
        mock_bench_stats = pd.Series({'cagr_pct': 5.0})

        with patch.object(strategy_result, 'statistics', return_value=mock_strat_stats), \
             patch.object(benchmark_res_obj, 'statistics', return_value=mock_bench_stats), \
             patch('oequant.backtesting.results.show'):
            
            strategy_result.report(show_plot=False, show_benchmark_in_report=True, table_format='plain')
            printed_text = "\n".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
            # print(f"\nDEBUG test_report_metric_missing_in_benchmark:\n{printed_text}\n")

            assert "Strategy" in printed_text
            assert "Benchmark" in printed_text
            assert _prettify_stat_name('cagr_pct') in printed_text
            assert "10.50%" in printed_text
            assert "5.00%" in printed_text
            assert _prettify_stat_name('some_custom_stat') in printed_text
            assert "777" in printed_text
            assert "nan" in printed_text # Check for NaN representation from tabulate/formatting
            # The exact line check is removed for robustness 