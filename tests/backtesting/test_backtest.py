import pandas as pd
import numpy as np
import pytest
from oequant.backtesting import backtest, BacktestResult
from oequant.data import get_data # Assuming get_data is robust enough for test data generation

# Helper to create sample OHLCV data
def create_sample_data(num_days=100, start_price=100.0, trend=0.01, volatility=0.02, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='B')
    prices = np.zeros(num_days)
    prices[0] = start_price
    for i in range(1, num_days):
        prices[i] = prices[i-1] * (1 + np.random.normal(loc=trend/252, scale=volatility/np.sqrt(252)))
    
    df = pd.DataFrame(index=dates)
    df['open'] = prices # Simplified: open=high=low=close for some tests
    df['high'] = prices
    df['low'] = prices
    df['close'] = prices
    df['volume'] = np.random.randint(1000, 10000, size=num_days)
    return df

@pytest.fixture
def sample_ohlcv_data():
    data = create_sample_data(num_days=20)
    data['entry'] = False
    data['exit'] = False
    return data

class TestBacktester:
    def test_no_trades(self, sample_ohlcv_data):
        data = sample_ohlcv_data
        result = backtest(data, entry_column='entry', exit_column='exit', capital=10000)
        assert isinstance(result, BacktestResult)
        assert result.trades.empty
        assert len(result.returns) == len(data)
        assert (result.returns['equity'] == 10000).all()
        assert result.initial_capital == 10000
        assert result.final_equity == 10000
        assert (result.returns['position_unit'] == 0).all()

    def test_one_trade_no_fees(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        exit_price = data['close'].iloc[5]

        result = backtest(data, entry_column='entry', exit_column='exit', size=10, size_unit='quantity', capital=10000)
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['trade_number'] == 1
        assert trade['entry_time'] == data.index[2]
        assert trade['exit_time'] == data.index[5]
        assert trade['entry_price'] == pytest.approx(entry_price)
        assert trade['exit_price'] == pytest.approx(exit_price)
        assert trade['quantity'] == 10
        assert trade['bars_held'] == 3 # 5 - 2
        assert trade['fee_total_currency'] == pytest.approx(0)
        expected_pnl_gross = (exit_price - entry_price) * 10
        assert trade['pnl_gross_currency'] == pytest.approx(expected_pnl_gross)
        assert trade['pnl_net_currency'] == pytest.approx(expected_pnl_gross)
        assert result.final_equity == pytest.approx(10000 + expected_pnl_gross)

        # Check returns dataframe for this trade period
        # Bar 2 (entry): equity starts at 10000. Net PnL for bar 2 (entry bar) from (close - entry_price)
        # (assuming entry_price_col is 'close', and signal_price_col is 'close')
        # For this simple test, entry_price is close[2], signal_price is close[2], so bar 2 pnl is 0.
        assert result.returns['return_net_currency'].iloc[2] == pytest.approx(0) 
        assert result.returns['equity'].iloc[2] == pytest.approx(10000) 
        assert result.returns['position_unit'].iloc[2] == 10

        # Bar 3 (hold): pnl = 10 * (close[3] - close[2])
        pnl_bar3 = 10 * (data['close'].iloc[3] - data['close'].iloc[2])
        assert result.returns['return_net_currency'].iloc[3] == pytest.approx(pnl_bar3)
        assert result.returns['equity'].iloc[3] == pytest.approx(10000 + pnl_bar3)
        assert result.returns['position_unit'].iloc[3] == 10

        # Bar 4 (hold): pnl = 10 * (close[4] - close[3])
        pnl_bar4 = 10 * (data['close'].iloc[4] - data['close'].iloc[3])
        assert result.returns['return_net_currency'].iloc[4] == pytest.approx(pnl_bar4)
        assert result.returns['equity'].iloc[4] == pytest.approx(10000 + pnl_bar3 + pnl_bar4)
        assert result.returns['position_unit'].iloc[4] == 10
        
        # Bar 5 (exit): pnl = 10 * (exit_price - close[4])
        pnl_bar5 = 10 * (exit_price - data['close'].iloc[4])
        assert result.returns['return_net_currency'].iloc[5] == pytest.approx(pnl_bar5)
        assert result.returns['equity'].iloc[5] == pytest.approx(10000 + pnl_bar3 + pnl_bar4 + pnl_bar5) # which is 10000 + total_pnl
        assert result.returns['position_unit'].iloc[5] == 0 # Position is 0 AFTER exit action on this bar

        # Equity after trade should be capital + PNL
        assert result.returns['equity'].iloc[-1] == pytest.approx(10000 + expected_pnl_gross)

    def test_one_trade_with_fractional_fee(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        exit_price = data['close'].iloc[5]
        fee_frac = 0.001 # 0.1%
        qty = 10

        result = backtest(data, entry_column='entry', exit_column='exit', size=qty, size_unit='quantity', capital=10000, fee_frac=fee_frac)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]

        entry_cost = entry_price * qty
        entry_fee = entry_cost * fee_frac
        exit_value = exit_price * qty
        exit_fee = exit_value * fee_frac
        total_fees = entry_fee + exit_fee

        assert trade['fee_total_currency'] == pytest.approx(total_fees)
        expected_pnl_gross = (exit_price - entry_price) * qty
        expected_pnl_net = expected_pnl_gross - total_fees
        assert trade['pnl_gross_currency'] == pytest.approx(expected_pnl_gross)
        assert trade['pnl_net_currency'] == pytest.approx(expected_pnl_net)
        assert result.final_equity == pytest.approx(10000 + expected_pnl_net)

        # Equity on entry bar should reflect entry fee
        # PnL on entry bar (close[2]-entry_price[2])*qty = 0. Net PnL = 0 - entry_fee.
        assert result.returns['return_net_currency'].iloc[2] == pytest.approx(-entry_fee)
        assert result.returns['equity'].iloc[2] == pytest.approx(10000 - entry_fee)

        # Equity on exit bar: equity before exit + MTM for bar - exit_fee
        equity_before_exit_bar_action = result.returns['equity'].iloc[4]
        mtm_exit_bar = qty * (exit_price - data['close'].iloc[4])
        assert result.returns['return_net_currency'].iloc[5] == pytest.approx(mtm_exit_bar - exit_fee)
        assert result.returns['equity'].iloc[5] == pytest.approx(equity_before_exit_bar_action + mtm_exit_bar - exit_fee)

    def test_one_trade_with_currency_fee(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        exit_price = data['close'].iloc[5]
        fee_curr = 0.5 # $0.5 per unit
        qty = 10

        result = backtest(data, entry_column='entry', exit_column='exit', size=qty, size_unit='quantity', capital=10000, fee_curr=fee_curr)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]

        entry_fee = qty * fee_curr
        exit_fee = qty * fee_curr
        total_fees = entry_fee + exit_fee

        assert trade['fee_total_currency'] == pytest.approx(total_fees)
        expected_pnl_gross = (exit_price - entry_price) * qty
        expected_pnl_net = expected_pnl_gross - total_fees
        assert trade['pnl_net_currency'] == pytest.approx(expected_pnl_net)
        assert result.final_equity == pytest.approx(10000 + expected_pnl_net)

    def test_non_fractional_positions(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        result = backtest(data, entry_column='entry', exit_column='exit', size=10.7, size_unit='quantity', capital=10000, allow_fractional_positions=False)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['quantity'] == 10 # Floor of 10.7

    def test_size_from_column(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data['trade_size'] = 15.0
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        
        result = backtest(data, entry_column='entry', exit_column='exit', size='trade_size', size_unit='quantity', capital=10000)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['quantity'] == 15.0

    def test_open_trade_at_end(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[15], 'entry'] = True # Trade opens near end, remains open
        entry_price = data['close'].iloc[15]
        last_price = data['close'].iloc[-1]
        qty = 5

        result = backtest(data, entry_column='entry', exit_column='exit', size=qty, size_unit='quantity', capital=10000)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade['entry_time'] == data.index[15]
        assert trade['exit_time'] == data.index[-1] # MTM close
        assert trade['exit_price'] == pytest.approx(last_price) # MTM exit price
        assert trade['fee_total_currency'] == pytest.approx(0) # No exit fees for MTM close
        expected_pnl_gross = (last_price - entry_price) * qty
        assert trade['pnl_gross_currency'] == pytest.approx(expected_pnl_gross)
        assert trade['pnl_net_currency'] == pytest.approx(expected_pnl_gross)
        assert result.final_equity == pytest.approx(10000 + expected_pnl_gross)
        assert result.returns['position_unit'].iloc[-1] == qty # Position held at last bar

    def test_entry_exit_same_bar(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[2], 'exit'] = True # Exit on the same bar as entry
        
        # Logic: exit processed first. So if in_trade is false, entry on same bar will happen.
        # If in_trade is true, exit will happen. Then entry for SAME BAR is skipped if not in_trade is false.
        # Current loop: exit, then entry. If exit happens, in_trade=False. Then entry happens.
        # This implies a day trade: enter and exit on the same bar.
        
        entry_price = data['close'].iloc[2]
        exit_price = data['close'].iloc[2] # exit_price_col defaults to 'close'
        qty = 10

        result = backtest(data, entry_column='entry', exit_column='exit', size=qty, size_unit='quantity', capital=10000)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]

        assert trade['entry_time'] == data.index[2]
        assert trade['exit_time'] == data.index[2]
        assert trade['bars_held'] == 0
        assert trade['pnl_gross_currency'] == pytest.approx(0) # (exit_price - entry_price) * qty = 0
        assert trade['pnl_net_currency'] == pytest.approx(0)
        assert result.final_equity == pytest.approx(10000)

        # Bar 2: entry, MTM (close-entry)=0. exit, MTM (exit-entry_for_bar_pnl_calc)=0. Total PNL for bar = 0.
        assert result.returns['return_net_currency'].iloc[2] == pytest.approx(0)
        assert result.returns['equity'].iloc[2] == pytest.approx(10000)
        assert result.returns['position_unit'].iloc[2] == 0 # Net position after exit

    def test_signal_price_col_different_from_entry_exit(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data['custom_mtm'] = data['close'] * 1.01 # A different price for MTM
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        
        entry_price = data['open'].iloc[2] # Entry at open
        exit_price = data['open'].iloc[5]  # Exit at open
        qty = 10

        result = backtest(data, entry_column='entry', exit_column='exit', 
                          entry_price_col='open', exit_price_col='open', 
                          signal_price_col='custom_mtm', 
                          size=qty, size_unit='quantity', capital=10000)
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_pnl_gross = (exit_price - entry_price) * qty
        assert trade['pnl_gross_currency'] == pytest.approx(expected_pnl_gross)

        # Check MTM based on 'custom_mtm'
        # Bar 2 (entry): gross_pnl_bar = qty * (custom_mtm[2] - open[2])
        pnl_bar2 = qty * (data['custom_mtm'].iloc[2] - data['open'].iloc[2])
        assert result.returns['return_gross_currency'].iloc[2] == pytest.approx(pnl_bar2)
        assert result.returns['equity'].iloc[2] == pytest.approx(10000 + pnl_bar2)

        # Bar 3 (hold): gross_pnl_bar = qty * (custom_mtm[3] - custom_mtm[2])
        pnl_bar3 = qty * (data['custom_mtm'].iloc[3] - data['custom_mtm'].iloc[2])
        assert result.returns['return_gross_currency'].iloc[3] == pytest.approx(pnl_bar3)
        assert result.returns['equity'].iloc[3] == pytest.approx(10000 + pnl_bar2 + pnl_bar3) 

    def test_size_unit_fraction_default_full_equity(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        initial_capital = 10000.0

        # size=None, size_unit='fraction' (default) should use 100% equity
        result = backtest(data, entry_column='entry', exit_column='exit',
                          capital=initial_capital, entry_price_col='close') 
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity = initial_capital / entry_price
        assert trade['quantity'] == pytest.approx(expected_quantity)
        assert trade['entry_price'] == pytest.approx(entry_price)

    def test_size_unit_fraction_specific_half_equity(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        initial_capital = 10000.0

        result = backtest(data, entry_column='entry', exit_column='exit', 
                          size=0.5, size_unit='fraction', capital=initial_capital, entry_price_col='close')
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity = (initial_capital * 0.5) / entry_price
        assert trade['quantity'] == pytest.approx(expected_quantity)

    def test_size_unit_fraction_column(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data['frac_size'] = 0.25 # Use 25% equity
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        initial_capital = 10000.0

        result = backtest(data, entry_column='entry', exit_column='exit', 
                          size='frac_size', size_unit='fraction', capital=initial_capital, entry_price_col='close')
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity = (initial_capital * 0.25) / entry_price
        assert trade['quantity'] == pytest.approx(expected_quantity)

    def test_size_unit_quantity_fixed(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        fixed_qty = 7
        result = backtest(data, entry_column='entry', exit_column='exit', 
                          size=fixed_qty, size_unit='quantity', capital=10000)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]['quantity'] == fixed_qty

    def test_size_unit_quantity_column(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data['qty_col'] = 8
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        result = backtest(data, entry_column='entry', exit_column='exit', 
                          size='qty_col', size_unit='quantity', capital=10000)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]['quantity'] == 8

    def test_size_unit_nominal_fixed(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]
        nominal_value = 5000.0

        result = backtest(data, entry_column='entry', exit_column='exit', 
                          size=nominal_value, size_unit='nominal', capital=10000, entry_price_col='close')
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity = nominal_value / entry_price
        assert trade['quantity'] == pytest.approx(expected_quantity)

    def test_size_unit_nominal_column(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data['nominal_col'] = 2500.0
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[5], 'exit'] = True
        entry_price = data['close'].iloc[2]

        result = backtest(data, entry_column='entry', exit_column='exit', 
                          size='nominal_col', size_unit='nominal', capital=10000, entry_price_col='close')
        
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity = data['nominal_col'].iloc[2] / entry_price
        assert trade['quantity'] == pytest.approx(expected_quantity)

    def test_size_unit_fraction_non_fractional_shares(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        entry_price = data['close'].iloc[2] # Approx 100
        initial_capital = 1050.0 # Should allow ~10.5 shares if fractional
        
        # With 100% equity (size=None or 1.0), quantity should be floor(1050/entry_price)
        result = backtest(data, entry_column='entry', exit_column='exit',
                          size=1.0, size_unit='fraction', capital=initial_capital, 
                          allow_fractional_positions=False, entry_price_col='close')
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity_calc = (initial_capital * 1.0) / entry_price
        assert trade['quantity'] == np.floor(expected_quantity_calc)
        assert trade['quantity'] < expected_quantity_calc # Ensure flooring happened if not whole number

    def test_size_unit_nominal_non_fractional_shares(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        entry_price = data['close'].iloc[2] # Approx 100
        nominal_value = 1070.0 # Should allow ~10.7 shares if fractional

        result = backtest(data, entry_column='entry', exit_column='exit',
                          size=nominal_value, size_unit='nominal', capital=10000, 
                          allow_fractional_positions=False, entry_price_col='close')
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        expected_quantity_calc = nominal_value / entry_price
        assert trade['quantity'] == np.floor(expected_quantity_calc)
        assert trade['quantity'] < expected_quantity_calc # Ensure flooring happened

    def test_size_zero_prevents_trade(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        result_qty = backtest(data, entry_column='entry', exit_column='exit', size=0, size_unit='quantity')
        assert result_qty.trades.empty

        data['zero_size_col'] = 0.0
        result_frac_col = backtest(data, entry_column='entry', exit_column='exit', size='zero_size_col', size_unit='fraction')
        assert result_frac_col.trades.empty

        result_nom = backtest(data, entry_column='entry', exit_column='exit', size=0, size_unit='nominal')
        assert result_nom.trades.empty

    def test_invalid_size_unit(self, sample_ohlcv_data):
        data = sample_ohlcv_data
        with pytest.raises(ValueError, match="Invalid size_unit: 'wrong_unit'. Must be 'fraction', 'quantity', or 'nominal'."):
            backtest(data, entry_column='entry', exit_column='exit', size_unit='wrong_unit')

    def test_entry_price_zero_for_fraction_or_nominal(self, sample_ohlcv_data):
        data = sample_ohlcv_data.copy()
        data.loc[data.index[2], 'entry'] = True
        data.loc[data.index[2], 'close'] = 0.0 # Set entry price to zero

        # Fraction sizing should result in 0 quantity if price is 0
        result_frac = backtest(data, entry_column='entry', exit_column='exit', 
                               size=0.5, size_unit='fraction', entry_price_col='close')
        assert result_frac.trades.empty # No trade should occur

        # Nominal sizing should result in 0 quantity if price is 0
        result_nom = backtest(data, entry_column='entry', exit_column='exit', 
                              size=1000, size_unit='nominal', entry_price_col='close')
        assert result_nom.trades.empty # No trade should occur 