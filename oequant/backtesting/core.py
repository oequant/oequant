import pandas as pd
import numpy as np
import math
from .results import BacktestResult

class Backtester:
    def __init__(self):
        # Potentially for global settings or pre-calculated data if needed later
        pass

    def backtest(
        self,
        data: pd.DataFrame,
        entry_column: str,
        exit_column: str,
        entry_price_col: str = 'close',
        exit_price_col: str = 'close',
        size_col_or_val: any = 1.0, # float for fixed units, str for column name
        fee_frac: float = 0.0,
        fee_curr: float = 0.0, # Static currency fee per unit
        capital: float = 100_000.0,
        allow_fractional_positions: bool = True,
        signal_price_col: str = 'close' # Column used for MTM calculations
    ) -> BacktestResult:
        """
        Core backtesting logic.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")

        # Validate required columns
        required_cols = {entry_column, exit_column, entry_price_col, exit_price_col, signal_price_col}
        if isinstance(size_col_or_val, str):
            required_cols.add(size_col_or_val)
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        # Initialize results storage
        trades_list = []
        num_bars = len(data)
        # .returns dataframe columns
        # ['equity', 'position_unit', 'position_usd', 'return_gross_frac', 'return_net_frac', 'return_gross_currency', 'return_net_currency']
        returns_data = np.full((num_bars, 7), np.nan)
        
        current_equity = capital
        in_trade = False
        current_position_units = 0.0
        trade_entry_price_actual = 0.0
        trade_cost_basis_total = 0.0 # Cost basis for the current trade (qty * entry_price_actual)
        trade_start_bar_index = -1
        trade_number = 0
        equity_at_trade_entry = capital

        for i in range(num_bars):
            bar_data = data.iloc[i]
            bar_time = data.index[i]
            equity_at_bar_start = current_equity
            
            bar_pnl_gross_curr = 0.0
            bar_fees_curr = 0.0

            # Process exits first, then entries for the same bar
            if in_trade and bar_data[exit_column]:
                exit_price_actual = bar_data[exit_price_col]
                proceeds_gross = current_position_units * exit_price_actual
                
                fee_exit_frac_val = proceeds_gross * fee_frac
                fee_exit_curr_val = current_position_units * fee_curr
                total_exit_fees = fee_exit_frac_val + fee_exit_curr_val
                bar_fees_curr += total_exit_fees

                # PnL from previous close to this exit price
                prev_signal_price = data[signal_price_col].iloc[i-1] if i > 0 else trade_entry_price_actual
                bar_pnl_gross_curr += current_position_units * (exit_price_actual - prev_signal_price)

                # Trade PnL calculations
                pnl_gross_trade_curr = proceeds_gross - trade_cost_basis_total
                total_trade_fees = trades_list[-1]['entry_fee_total_currency'] + total_exit_fees # entry_fee already in trades_list for current trade
                pnl_net_trade_curr = pnl_gross_trade_curr - total_trade_fees

                trades_list[-1].update({
                    'exit_time': bar_time,
                    'exit_price': exit_price_actual,
                    'bars_held': i - trade_start_bar_index,
                    'exit_fee_frac_value': fee_exit_frac_val,
                    'exit_fee_curr_value': fee_exit_curr_val,
                    'fee_total_currency': total_trade_fees, 
                    'fee_total_as_fraction_of_equity': total_trade_fees / equity_at_trade_entry if equity_at_trade_entry else 0,
                    'pnl_gross_currency': pnl_gross_trade_curr,
                    'pnl_gross_frac': pnl_gross_trade_curr / trade_cost_basis_total if trade_cost_basis_total else 0,
                    'pnl_net_currency': pnl_net_trade_curr,
                    'pnl_net_frac': pnl_net_trade_curr / trade_cost_basis_total if trade_cost_basis_total else 0,
                })
                
                in_trade = False
                current_position_units = 0.0
                # current_equity is updated by bar_pnl_net_curr at the end of bar processing

            if not in_trade and bar_data[entry_column]:
                trade_number += 1
                trade_entry_price_actual = bar_data[entry_price_col]
                
                if isinstance(size_col_or_val, str):
                    quantity_requested = bar_data[size_col_or_val]
                else:
                    quantity_requested = float(size_col_or_val)
                
                if not allow_fractional_positions:
                    quantity = math.floor(quantity_requested)
                else:
                    quantity = quantity_requested
                
                if quantity <= 0:
                    continue # Skip if trying to enter with zero or negative size

                trade_cost_basis_total = quantity * trade_entry_price_actual
                fee_entry_frac_val = trade_cost_basis_total * fee_frac
                fee_entry_curr_val = quantity * fee_curr
                total_entry_fees = fee_entry_frac_val + fee_entry_curr_val
                bar_fees_curr += total_entry_fees

                # PnL from entry to this bar's signal price
                current_signal_price = bar_data[signal_price_col]
                bar_pnl_gross_curr += quantity * (current_signal_price - trade_entry_price_actual)
                
                current_position_units = quantity
                in_trade = True
                trade_start_bar_index = i
                equity_at_trade_entry = current_equity # Equity before this trade's entry fees are deducted

                trades_list.append({
                    'trade_number': trade_number,
                    'entry_time': bar_time,
                    'entry_price': trade_entry_price_actual,
                    'quantity': quantity,
                    'entry_fee_frac_value': fee_entry_frac_val,
                    'entry_fee_curr_value': fee_entry_curr_val,
                    'entry_fee_total_currency': total_entry_fees, # Store entry fee part
                    # Exit fields will be filled on exit
                    'exit_time': pd.NaT, 'exit_price': np.nan, 'bars_held': 0,
                    'exit_fee_frac_value': np.nan, 'exit_fee_curr_value': np.nan,
                    'fee_total_currency': np.nan, 'fee_total_as_fraction_of_equity': np.nan,
                    'pnl_gross_currency': np.nan, 'pnl_gross_frac': np.nan,
                    'pnl_net_currency': np.nan, 'pnl_net_frac': np.nan,
                })

                # Check for immediate exit on the same bar
                if trade_start_bar_index == i and bar_data[exit_column]: # A trade was just opened and exit signal is also true
                    exit_price_actual_immediate = bar_data[exit_price_col]
                    proceeds_gross_immediate = current_position_units * exit_price_actual_immediate

                    fee_exit_frac_val_immediate = proceeds_gross_immediate * fee_frac
                    fee_exit_curr_val_immediate = current_position_units * fee_curr
                    total_exit_fees_immediate = fee_exit_frac_val_immediate + fee_exit_curr_val_immediate
                    bar_fees_curr += total_exit_fees_immediate # Add to this bar's fees

                    # Adjust bar_pnl_gross_curr: it already has (signal - entry_price) component.
                    # Add (exit_price - signal_price) component.
                    bar_pnl_gross_curr += current_position_units * (exit_price_actual_immediate - bar_data[signal_price_col])

                    # Update the trade record for immediate exit
                    pnl_gross_trade_curr_immediate = proceeds_gross_immediate - trade_cost_basis_total
                    entry_fees_for_this_trade = trades_list[-1]['entry_fee_total_currency'] # Already recorded
                    total_trade_fees_immediate = entry_fees_for_this_trade + total_exit_fees_immediate
                    pnl_net_trade_curr_immediate = pnl_gross_trade_curr_immediate - total_trade_fees_immediate

                    trades_list[-1].update({
                        'exit_time': bar_time,
                        'exit_price': exit_price_actual_immediate,
                        'bars_held': 0, # Exited on the same bar
                        'exit_fee_frac_value': fee_exit_frac_val_immediate,
                        'exit_fee_curr_value': fee_exit_curr_val_immediate,
                        'fee_total_currency': total_trade_fees_immediate,
                        'fee_total_as_fraction_of_equity': total_trade_fees_immediate / equity_at_trade_entry if equity_at_trade_entry else 0,
                        'pnl_gross_currency': pnl_gross_trade_curr_immediate,
                        'pnl_gross_frac': pnl_gross_trade_curr_immediate / trade_cost_basis_total if trade_cost_basis_total else 0,
                        'pnl_net_currency': pnl_net_trade_curr_immediate,
                        'pnl_net_frac': pnl_net_trade_curr_immediate / trade_cost_basis_total if trade_cost_basis_total else 0,
                    })
                    in_trade = False # Exited
                    current_position_units = 0.0
            
            elif in_trade: # Holding a position (that wasn't immediately exited after entry)
                current_signal_price = bar_data[signal_price_col]
                prev_signal_price = data[signal_price_col].iloc[i-1] if i > 0 else trade_entry_price_actual
                bar_pnl_gross_curr += current_position_units * (current_signal_price - prev_signal_price)

            bar_pnl_net_curr = bar_pnl_gross_curr - bar_fees_curr
            current_equity = equity_at_bar_start + bar_pnl_net_curr # This is the key equity update line

            position_units_at_bar_end = current_position_units # Position held *into* next bar
            position_usd_at_bar_end = position_units_at_bar_end * bar_data[signal_price_col]

            returns_data[i, 0] = current_equity
            returns_data[i, 1] = position_units_at_bar_end
            returns_data[i, 2] = position_usd_at_bar_end
            returns_data[i, 3] = bar_pnl_gross_curr / equity_at_bar_start if equity_at_bar_start != 0 else 0
            returns_data[i, 4] = bar_pnl_net_curr / equity_at_bar_start if equity_at_bar_start != 0 else 0
            returns_data[i, 5] = bar_pnl_gross_curr
            returns_data[i, 6] = bar_pnl_net_curr
        
        # Finalize trades that are still open at the end of data
        if in_trade:
            # Mark to market close for the last bar
            last_bar_data = data.iloc[-1]
            last_price = last_bar_data[signal_price_col] # Use signal_price_col for MTM close
            proceeds_gross = current_position_units * last_price
            
            # No exit fees for MTM close unless specified, assume 0 for now
            total_exit_fees = 0.0 
            # bar_fees_curr += total_exit_fees # Not part of last bar's fees, part of trade itself

            pnl_gross_trade_curr = proceeds_gross - trade_cost_basis_total
            total_trade_fees = trades_list[-1]['entry_fee_total_currency'] + total_exit_fees
            pnl_net_trade_curr = pnl_gross_trade_curr - total_trade_fees

            trades_list[-1].update({
                'exit_time': data.index[-1], # Mark as exited at last bar time
                'exit_price': last_price, # MTM exit price
                'bars_held': num_bars - 1 - trade_start_bar_index,
                'exit_fee_frac_value': 0.0, # No actual exit fee
                'exit_fee_curr_value': 0.0, # No actual exit fee
                'fee_total_currency': total_trade_fees, 
                'fee_total_as_fraction_of_equity': total_trade_fees / equity_at_trade_entry if equity_at_trade_entry else 0,
                'pnl_gross_currency': pnl_gross_trade_curr,
                'pnl_gross_frac': pnl_gross_trade_curr / trade_cost_basis_total if trade_cost_basis_total else 0,
                'pnl_net_currency': pnl_net_trade_curr,
                'pnl_net_frac': pnl_net_trade_curr / trade_cost_basis_total if trade_cost_basis_total else 0,
            })
            # Equity is already MTM by the loop

        returns_df_cols = ['equity', 'position_unit', 'position_usd', 'return_gross_frac', 'return_net_frac', 'return_gross_currency', 'return_net_currency']
        returns_df = pd.DataFrame(returns_data, index=data.index, columns=returns_df_cols)
        trades_df = pd.DataFrame(trades_list)
        
        # Ensure correct dtypes for trades_df, especially for times and numeric
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            for col in ['entry_price', 'quantity', 'entry_fee_frac_value', 'entry_fee_curr_value', 
                        'entry_fee_total_currency', 'exit_price', 'bars_held', 'exit_fee_frac_value', 
                        'exit_fee_curr_value', 'fee_total_currency', 'fee_total_as_fraction_of_equity',
                        'pnl_gross_currency', 'pnl_gross_frac', 'pnl_net_currency', 'pnl_net_frac']:
                if col in trades_df.columns:
                     trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')

        return BacktestResult(trades=trades_df, returns=returns_df, initial_capital=capital, final_equity=current_equity, ohlcv_data=data)


def backtest(
    data: pd.DataFrame,
    entry_column: str,
    exit_column: str,
    entry_price_col: str = 'close',
    exit_price_col: str = 'close',
    size_col_or_val: any = 1.0,
    fee_frac: float = 0.0,
    fee_curr: float = 0.0,
    capital: float = 100_000.0,
    allow_fractional_positions: bool = True,
    signal_price_col: str = 'close'
) -> BacktestResult:
    """
    User-friendly functional wrapper for the Backtester class.
    """
    engine = Backtester()
    return engine.backtest(
        data=data,
        entry_column=entry_column,
        exit_column=exit_column,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        size_col_or_val=size_col_or_val,
        fee_frac=fee_frac,
        fee_curr=fee_curr,
        capital=capital,
        allow_fractional_positions=allow_fractional_positions,
        signal_price_col=signal_price_col
    ) 