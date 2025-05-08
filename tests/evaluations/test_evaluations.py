import pandas as pd
import numpy as np
import pytest
from oequant.backtesting.results import BacktestResult
from oequant.evaluations import calculate_statistics

@pytest.fixture
def sample_backtest_result_no_trades():
    num_days = 10
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='B')
    returns_data = {
        'equity': [10000.0] * num_days,
        'position_unit': [0.0] * num_days,
        'position_usd': [0.0] * num_days,
        'return_gross_frac': [0.0] * num_days,
        'return_net_frac': [0.0] * num_days,
        'return_gross_currency': [0.0] * num_days,
        'return_net_currency': [0.0] * num_days,
    }
    returns_df = pd.DataFrame(returns_data, index=dates)
    trades_df = pd.DataFrame()
    # Create dummy ohlcv_data
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_df = pd.DataFrame(100, index=dates, columns=ohlcv_columns)
    return BacktestResult(trades=trades_df, returns=returns_df, ohlcv_data=ohlcv_df, initial_capital=10000.0, final_equity=10000.0)

@pytest.fixture
def sample_backtest_result_one_trade():
    # Data for a 5-day period, trade from day 1 to day 3 (inclusive of start day, exit on day 3 open)
    dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
    initial_capital = 10000.0
    # Entry: Day 1 at 100, Qty: 10. Cost: 1000. Fee: 1 (net entry cost 1001)
    # Day 1 close: 101. MTM PnL: 10*(101-100)=10. Equity: 10000 - 1 (fee) + 10 = 10009
    # Day 2 close: 102. MTM PnL: 10*(102-101)=10. Equity: 10009 + 10 = 10019
    # Day 3 exit: 103. MTM PnL for exit bar: 10*(103-102)=10. Fee: 1.03 (10*103*0.0001 -> simplified to 1 for test). Equity: 10019 + 10 - 1(fee) = 10028
    # Final equity = 10028
    # Trade PNL: Gross: 10*(103-100)=30. Net: 30 - 1 (entry_fee) - 1 (exit_fee) = 28

    returns_data = {
        'equity':                [initial_capital, 10000.0 - 1.0 + 10, 10009.0 + 10, 10019.0 + 10 - 1.03, 10027.97], # Day 0 is pre-trade
        'position_unit':         [0, 10, 10, 0, 0],
        'position_usd':          [0, 10*101, 10*102, 0, 0],
        'return_gross_frac':     [0, 10/initial_capital,   10/10009.0,         10/10019.0, 0],
        'return_net_frac':       [0, (10-1.0)/initial_capital, 10/10009.0,   (10-1.03)/10019.0, 0],
        'return_gross_currency': [0, 10, 10, 10, 0],
        'return_net_currency':   [0, 10-1.0, 10, 10-1.03, 0]
    }
    # Adjust Day 0 to have initial capital and 0 returns
    returns_data['equity'][0] = initial_capital
    returns_data['return_gross_frac'][0] = 0.0
    returns_data['return_net_frac'][0] = 0.0
    returns_data['return_gross_currency'][0] = 0.0
    returns_data['return_net_currency'][0] = 0.0

    returns_df = pd.DataFrame(returns_data, index=dates)

    trades_data = [{
        'trade_number': 1, 'entry_time': dates[1], 'entry_price': 100.0, 'quantity': 10,
        'entry_fee_frac_value': 0, 'entry_fee_curr_value': 1.0, 'entry_fee_total_currency':1.0,
        'exit_time': dates[3], 'exit_price': 103.0, 'bars_held': 2,
        'exit_fee_frac_value': 0, 'exit_fee_curr_value': 1.03, 'fee_total_currency': 2.03,
        'fee_total_as_fraction_of_equity': 2.03/initial_capital,
        'pnl_gross_currency': 30.0, 'pnl_gross_frac': 30.0/1000.0,
        'pnl_net_currency': 30.0 - 2.03, 'pnl_net_frac': (30.0-2.03)/1000.0
    }]
    trades_df = pd.DataFrame(trades_data)
    final_equity = returns_df['equity'].iloc[-1]
    # Create dummy ohlcv_data
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_df = pd.DataFrame(100, index=dates, columns=ohlcv_columns)
    return BacktestResult(trades=trades_df, returns=returns_df, ohlcv_data=ohlcv_df, initial_capital=initial_capital, final_equity=final_equity)


class TestEvaluations:
    def test_calculate_statistics_no_trades(self, sample_backtest_result_no_trades):
        stats = calculate_statistics(sample_backtest_result_no_trades)
        assert stats['cagr_pct'] == 0.0
        assert stats['return_per_trade_pct'] == 0.0
        assert np.isnan(stats['sharpe_ratio'])
        assert np.isnan(stats['sortino_ratio'])
        assert np.isnan(stats['serenity_ratio']) # Changed to isnan for no profit/no drawdown case
        assert stats['pct_in_position'] == 0.0
        assert stats['max_dd_pct'] == 0.0
        assert np.isnan(stats['cagr_to_max_dd'])
        assert stats['total_trades'] == 0
        assert stats['win_rate_pct'] == 0.0
        assert stats['profit_factor'] == 1.0 # Adjusted expectation for no PnL
        assert stats['avg_bars_held'] == 0.0 # Added assertion

    def test_calculate_statistics_one_trade_net(self, sample_backtest_result_one_trade):
        result = sample_backtest_result_one_trade
        stats = calculate_statistics(result, PnL_type='net')

        # CAGR
        # Total return = (10027.97 / 10000) - 1 = 0.002797
        # Years = 5 / 252 = 0.01984
        # CAGR = ((1 + 0.002797)**(1/0.01984) - 1)*100 = ((1.002797)**50.4 - 1)*100 approx 14.9%
        # My CAGR logic for < 1 year: total_return * 100
        expected_cagr = ((result.final_equity / result.initial_capital) - 1) * 100
        assert stats['cagr_pct'] == pytest.approx(expected_cagr, abs=0.1)
        
        assert stats['return_per_trade_pct'] == pytest.approx(((30.0-2.03)/1000.0)*100, abs=0.01) # (27.97/1000)*100 = 2.797
        
        # Sharpe and Sortino are complex to verify without full calc, check for non-NaN for now
        assert not np.isnan(stats['sharpe_ratio'])
        assert not np.isnan(stats['sortino_ratio'])

        # Pct in position: 2 bars with position_unit > 0 out of 5 bars (day 1, day 2) -> 2/5 = 40%
        assert stats['pct_in_position'] == pytest.approx((2/5)*100)

        # Max Drawdown
        # Equity: [10000, 10009, 10019, 10027.97, 10027.97] (equity includes current bar's PnL & fees)
        # Simplified for fixture: Equity at start of bar: [10000, 10000, 10009, 10019, 10027.97]
        # Returns equity is: [10000.0, 10009.0, 10019.0, 10027.97, 10027.97]
        # Rolling Max: [10000.0, 10009.0, 10019.0, 10027.97, 10027.97]
        # Drawdown: all zeros in this simplified winning case. Max DD = 0.
        # Let's refine fixture to have a drawdown.
        # If entry fee was higher, e.g. 10. Equity: [10000, 10000(start), 10000-10+10=10000, 10000+10=10010, 10010+10-1=10019]
        # Max DD would be 0 if returns are always positive or flat after fees. 
        # For this specific fixture, entry fee is 1, PNL on first day is 10. Net +9. Equity goes up.
        assert stats['max_dd_pct'] == pytest.approx(0.0, abs=0.001) # Expected to be 0 or very slightly negative from entry fee if equity dips
                                                                     # For the sample_backtest_result_one_trade, equity never drops below initial_capital after the first bar.
                                                                     # Let's check the equity curve for the fixture: 10000, 10009, 10019, 10027.97, 10027.97. No drawdown.

        assert stats['total_trades'] == 1
        assert stats['win_rate_pct'] == 100.0
        assert stats['profit_factor'] > 1 # Will be inf if no losses

        # Serenity Ratio: total_net_profit_currency / max_abs_dd_currency
        # Total net profit = 27.97. Max DD = 0. Should be inf.
        assert np.isinf(stats['serenity_ratio'])
        assert np.isinf(stats['cagr_to_max_dd']) # As max_dd_pct is 0
        assert stats['avg_bars_held'] == 2.0 # Trade in fixture has bars_held = 2

    def test_calculate_statistics_one_trade_gross(self, sample_backtest_result_one_trade):
        result = sample_backtest_result_one_trade
        # For gross, final_equity and initial_capital are the same, but internal calcs use gross returns.
        # We need a different BacktestResult or adjust the fixture for meaningful gross test.
        # For now, just test that it runs and some PnL_type dependent values change.
        stats_net = calculate_statistics(result, PnL_type='net')
        stats_gross = calculate_statistics(result, PnL_type='gross')

        assert stats_gross['return_per_trade_pct'] == pytest.approx((30.0/1000.0)*100, abs=0.01) # Gross PnL
        assert stats_gross['return_per_trade_pct'] > stats_net['return_per_trade_pct']
        # Sharpe/Sortino should also differ if returns differ
        if not (np.isnan(stats_net['sharpe_ratio']) or np.isnan(stats_gross['sharpe_ratio'])):
             assert stats_gross['sharpe_ratio'] != stats_net['sharpe_ratio']
        assert stats_gross['avg_bars_held'] == 2.0 # Should be the same as net for this metric

    def test_max_drawdown_calculation(self):
        dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
        initial_capital = 10000.0
        equity_values = [10000, 10100, 9900, 9800, 10200] # Peak 10100, Trough 9800 from this peak
        returns_data = {
            'equity': equity_values,
            'position_unit': [0,1,1,1,0],
            'return_net_frac': [0, 0.01, -0.0198, -0.0101, 0.0408] # Approx
        }
        # Fill other cols with zeros for simplicity
        for col in ['position_usd', 'return_gross_frac', 'return_gross_currency', 'return_net_currency']:
            returns_data[col] = [0.0] * 5
        returns_df = pd.DataFrame(returns_data, index=dates)
        trades_df = pd.DataFrame([{'pnl_net_currency': 200, 'pnl_net_frac': 0.02}]) # Dummy trade
        
        # Create dummy ohlcv_data
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_df = pd.DataFrame(100, index=dates, columns=ohlcv_columns)
        result = BacktestResult(trades=trades_df, returns=returns_df, ohlcv_data=ohlcv_df, initial_capital=initial_capital, final_equity=equity_values[-1])
        stats = calculate_statistics(result, PnL_type='net')
        
        # Rolling Max: [10000, 10100, 10100, 10100, 10200]
        # Drawdown values: [0, 0, (9900-10100)/10100, (9800-10100)/10100, 0]
        # Drawdown values: [0, 0, -0.01980198, -0.02970297, 0]
        # Max DD = -0.02970297
        assert stats['max_dd_pct'] == pytest.approx(-0.02970297 * 100, abs=0.01)

        # Serenity: Total Profit = 200. Max Abs DD = 0.02970297 * 10100 = 300
        # Serenity = 200 / 300 = 0.666
        assert stats['serenity_ratio'] == pytest.approx(200 / (0.02970297 * 10100), abs=0.01)
        # This test uses a dummy trades_df, need to add 'bars_held' for avg_bars_held
        # For now, if not present, it defaults to 0. Let's assert that, or update fixture later if specific value needed.
        # The calculate_statistics defaults to 0 if 'bars_held' is not present or trades_df is empty.
        # The dummy trade_df here is `trades_df = pd.DataFrame([{'pnl_net_currency': 200, 'pnl_net_frac': 0.02}])`
        # It doesn't have 'bars_held'. So, avg_bars_held should be 0.
        assert stats['avg_bars_held'] == 0.0

    def test_empty_returns_df(self):
        empty_returns = pd.DataFrame(columns=['equity', 'position_unit', 'position_usd', 'return_gross_frac', 'return_net_frac', 'return_gross_currency', 'return_net_currency'])
        empty_returns.index = pd.to_datetime(empty_returns.index)
        trades_df = pd.DataFrame()
        # Create dummy ohlcv_data
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_df = pd.DataFrame(index=empty_returns.index, columns=ohlcv_columns).fillna(100)
        result = BacktestResult(trades=trades_df, returns=empty_returns, ohlcv_data=ohlcv_df, initial_capital=10000, final_equity=10000)
        stats = calculate_statistics(result)
        assert stats['cagr_pct'] == 0.0
        assert np.isnan(stats['sharpe_ratio'])
        assert stats['total_trades'] == 0
        assert stats['avg_bars_held'] == 0.0 # Added assertion

