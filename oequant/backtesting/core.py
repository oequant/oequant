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
        size: any = None, # Renamed from size_col_or_val, default None to handle 'fraction' 1.0 internally
        size_unit: str = 'fraction', # 'fraction', 'quantity', 'nominal'
        fee_frac: float = 0.0,
        fee_curr: float = 0.0, # Static currency fee per unit
        capital: float = 100_000.0,
        allow_fractional_positions: bool = True,
        signal_price_col: str = 'close' # Column used for MTM calculations
    ) -> BacktestResult:
        """
        Core backtesting logic.
        Handles column names or pandas expressions for entry/exit signals.

        Args:
            data (pd.DataFrame): Input OHLCV data with a DatetimeIndex.
            entry_column (str): Column name or pandas expression for entry signals (boolean).
            exit_column (str): Column name or pandas expression for exit signals (boolean).
            entry_price_col (str, optional): Column for actual entry prices. Defaults to 'close'.
            exit_price_col (str, optional): Column for actual exit prices. Defaults to 'close'.
            size (any, optional): Size of the position.
                - If `size_unit` is 'fraction' (default):
                    - If `None` (default), uses 1.0 (100% of current equity).
                    - If float, specifies fraction of current equity (e.g., 0.5 for 50%).
                    - If str (column name), column contains the fraction of equity.
                - If `size_unit` is 'quantity':
                    - If float/int, specifies the number of units/shares.
                    - If str (column name), column contains the number of units/shares.
                - If `size_unit` is 'nominal':
                    - If float/int, specifies the nominal monetary value to invest.
                    - If str (column name), column contains the nominal monetary value.
            size_unit (str, optional): Unit for `size`. Can be 'fraction', 'quantity', or 'nominal'.
                Defaults to 'fraction'.
            fee_frac (float, optional): Fractional fee per trade (e.g., 0.001 for 0.1%). Defaults to 0.0.
            fee_curr (float, optional): Currency fee per unit traded. Defaults to 0.0.
            capital (float, optional): Initial capital. Defaults to 100,000.0.
            allow_fractional_positions (bool, optional): Whether to allow fractional shares/units.
                Defaults to True.
            signal_price_col (str, optional): Column for mark-to-market calculations. Defaults to 'close'.

        Returns:
            BacktestResult: Object containing trades, returns, and other results.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")
        
        # Work on a copy to avoid modifying original DataFrame
        data_copy = data.copy()
        
        # --- Evaluate Entry/Exit Expressions --- 
        internal_entry_col = entry_column
        internal_exit_col = exit_column
        temp_cols_to_drop = []

        if entry_column not in data_copy.columns:
            try:
                entry_signal = data_copy.eval(entry_column)
                if not pd.api.types.is_bool_dtype(entry_signal):
                    raise TypeError(f"Entry expression '{entry_column}' did not evaluate to boolean Series.")
                internal_entry_col = '__entry_signal__'
                data_copy[internal_entry_col] = entry_signal
                temp_cols_to_drop.append(internal_entry_col)
                print(f"Evaluated entry expression: '{entry_column}' -> {internal_entry_col}")
            except Exception as e:
                raise ValueError(f"Failed to evaluate entry expression '{entry_column}'. Original error: {e}") from e
                
        if exit_column not in data_copy.columns:
            try:
                exit_signal = data_copy.eval(exit_column)
                if not pd.api.types.is_bool_dtype(exit_signal):
                    raise TypeError(f"Exit expression '{exit_column}' did not evaluate to boolean Series.")
                internal_exit_col = '__exit_signal__'
                data_copy[internal_exit_col] = exit_signal
                temp_cols_to_drop.append(internal_exit_col)
                print(f"Evaluated exit expression: '{exit_column}' -> {internal_exit_col}")
            except Exception as e:
                raise ValueError(f"Failed to evaluate exit expression '{exit_column}'. Original error: {e}") from e
        # --- End Expression Evaluation ---

        # Validate required columns
        required_cols = {internal_entry_col, internal_exit_col, entry_price_col, exit_price_col, signal_price_col}
        if isinstance(size, str): # Check if size is a column name
            required_cols.add(size)
        missing_cols = required_cols - set(data_copy.columns)
        if missing_cols:
            data_copy.drop(columns=temp_cols_to_drop, inplace=True, errors='ignore')
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        if size_unit not in ['fraction', 'quantity', 'nominal']:
            data_copy.drop(columns=temp_cols_to_drop, inplace=True, errors='ignore')
            raise ValueError(f"Invalid size_unit: '{size_unit}'. Must be 'fraction', 'quantity', or 'nominal'.")

        # Initialize results storage
        trades_list = []
        num_bars = len(data_copy)
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

        # Use internal column names within the loop
        entry_signal_series = data_copy[internal_entry_col]
        exit_signal_series = data_copy[internal_exit_col]
        entry_price_series = data_copy[entry_price_col]
        exit_price_series = data_copy[exit_price_col]
        signal_price_series = data_copy[signal_price_col]
        
        # Handle size parameter
        size_is_column = isinstance(size, str)
        size_series_data = data_copy[size] if size_is_column else None
        
        # Effective size for 'fraction' unit if size is None (default)
        effective_size_val = size
        if size_unit == 'fraction' and size is None:
            effective_size_val = 1.0 # Default to 100% equity if size is None for fraction unit

        # --- Main Backtest Loop --- 
        # (Using optimized access where possible, e.g. pre-selected series)
        for i in range(num_bars):
            # bar_data = data_copy.iloc[i] # Less efficient than using series directly
            bar_time = data_copy.index[i]
            equity_at_bar_start = current_equity
            
            bar_pnl_gross_curr = 0.0
            bar_fees_curr = 0.0

            # Process exits first, then entries for the same bar
            exit_signal_active = exit_signal_series.iloc[i]
            if in_trade and exit_signal_active:
                exit_price_actual = exit_price_series.iloc[i]
                proceeds_gross = current_position_units * exit_price_actual
                
                fee_exit_frac_val = proceeds_gross * fee_frac
                fee_exit_curr_val = current_position_units * fee_curr
                total_exit_fees = fee_exit_frac_val + fee_exit_curr_val
                bar_fees_curr += total_exit_fees

                # PnL from previous close to this exit price
                prev_signal_price = signal_price_series.iloc[i-1] if i > 0 else trade_entry_price_actual
                bar_pnl_gross_curr += current_position_units * (exit_price_actual - prev_signal_price)

                # Trade PnL calculations
                pnl_gross_trade_curr = proceeds_gross - trade_cost_basis_total
                # Safely access entry fee assuming trades_list[-1] exists because in_trade is True
                total_trade_fees = trades_list[-1]['entry_fee_total_currency'] + total_exit_fees 
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

            # Process entry signal for this bar
            entry_signal_active = entry_signal_series.iloc[i]
            if not in_trade and entry_signal_active:
                trade_number += 1
                trade_entry_price_actual = entry_price_series.iloc[i]
                
                # --- Sizing Logic ---
                current_size_input = 0.0
                if size_is_column:
                    current_size_input = size_series_data.iloc[i]
                elif effective_size_val is not None: # Handles fixed numeric size or default fraction
                    current_size_input = float(effective_size_val)
                else: # Should not happen if logic is correct, but as a fallback
                    data_copy.drop(columns=temp_cols_to_drop, inplace=True, errors='ignore')
                    raise ValueError("Size parameter is not correctly defined for trade entry.")

                quantity_requested = 0.0
                if size_unit == 'fraction':
                    if trade_entry_price_actual <= 0: # Avoid division by zero or investing in valueless asset
                        quantity_requested = 0.0 
                    else:
                        # Use equity_at_bar_start as it reflects capital before this trade's costs
                        quantity_requested = (current_size_input * equity_at_bar_start) / trade_entry_price_actual
                elif size_unit == 'quantity':
                    quantity_requested = current_size_input
                elif size_unit == 'nominal':
                    if trade_entry_price_actual <= 0:
                        quantity_requested = 0.0
                    else:
                        quantity_requested = current_size_input / trade_entry_price_actual
                # --- End Sizing Logic ---
                
                if not allow_fractional_positions:
                    quantity = math.floor(quantity_requested)
                else:
                    quantity = quantity_requested
                
                if quantity <= 0:
                    # Skip trade if size is zero or negative, reset trade number if needed?
                    # Maybe log a warning? For now, just continue.
                    trade_number -= 1 # Decrement because trade didn't actually happen
                    continue 

                trade_cost_basis_total = quantity * trade_entry_price_actual
                fee_entry_frac_val = trade_cost_basis_total * fee_frac
                fee_entry_curr_val = quantity * fee_curr
                total_entry_fees = fee_entry_frac_val + fee_entry_curr_val
                bar_fees_curr += total_entry_fees

                # PnL from entry to this bar's signal price
                current_signal_price = signal_price_series.iloc[i]
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
                if trade_start_bar_index == i and exit_signal_series.iloc[i]: # A trade was just opened and exit signal is also true
                    exit_price_actual_immediate = exit_price_series.iloc[i]
                    proceeds_gross_immediate = current_position_units * exit_price_actual_immediate

                    fee_exit_frac_val_immediate = proceeds_gross_immediate * fee_frac
                    fee_exit_curr_val_immediate = current_position_units * fee_curr
                    total_exit_fees_immediate = fee_exit_frac_val_immediate + fee_exit_curr_val_immediate
                    bar_fees_curr += total_exit_fees_immediate # Add to this bar's fees

                    # Adjust bar_pnl_gross_curr: it already has (signal - entry_price) component.
                    # Add (exit_price - signal_price) component.
                    bar_pnl_gross_curr += current_position_units * (exit_price_actual_immediate - signal_price_series.iloc[i])

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
                current_signal_price = signal_price_series.iloc[i]
                prev_signal_price = signal_price_series.iloc[i-1] if i > 0 else trade_entry_price_actual
                bar_pnl_gross_curr += current_position_units * (current_signal_price - prev_signal_price)

            bar_pnl_net_curr = bar_pnl_gross_curr - bar_fees_curr
            current_equity = equity_at_bar_start + bar_pnl_net_curr # This is the key equity update line

            position_units_at_bar_end = current_position_units # Position held *into* next bar
            position_usd_at_bar_end = position_units_at_bar_end * signal_price_series.iloc[i]

            returns_data[i, 0] = current_equity
            returns_data[i, 1] = position_units_at_bar_end
            returns_data[i, 2] = position_usd_at_bar_end
            returns_data[i, 3] = bar_pnl_gross_curr / equity_at_bar_start if equity_at_bar_start != 0 else 0
            returns_data[i, 4] = bar_pnl_net_curr / equity_at_bar_start if equity_at_bar_start != 0 else 0
            returns_data[i, 5] = bar_pnl_gross_curr
            returns_data[i, 6] = bar_pnl_net_curr
        # --- End Backtest Loop --- 
        
        # Finalize trades that are still open at the end of data
        if in_trade:
            last_bar_data = data_copy.iloc[-1] # Need original bar data here for MTM price
            last_price = last_bar_data[signal_price_col] 
            proceeds_gross = current_position_units * last_price
            
            total_exit_fees = 0.0 # Assume no exit fees for MTM close

            pnl_gross_trade_curr = proceeds_gross - trade_cost_basis_total
            total_trade_fees = trades_list[-1]['entry_fee_total_currency'] + total_exit_fees
            pnl_net_trade_curr = pnl_gross_trade_curr - total_trade_fees

            trades_list[-1].update({
                'exit_time': data_copy.index[-1],
                'exit_price': last_price,
                'bars_held': num_bars - 1 - trade_start_bar_index,
                'exit_fee_frac_value': 0.0,
                'exit_fee_curr_value': 0.0,
                'fee_total_currency': total_trade_fees, 
                'fee_total_as_fraction_of_equity': total_trade_fees / equity_at_trade_entry if equity_at_trade_entry else 0,
                'pnl_gross_currency': pnl_gross_trade_curr,
                'pnl_gross_frac': pnl_gross_trade_curr / trade_cost_basis_total if trade_cost_basis_total else 0,
                'pnl_net_currency': pnl_net_trade_curr,
                'pnl_net_frac': pnl_net_trade_curr / trade_cost_basis_total if trade_cost_basis_total else 0,
            })

        returns_df_cols = ['equity', 'position_unit', 'position_usd', 'return_gross_frac', 'return_net_frac', 'return_gross_currency', 'return_net_currency']
        returns_df = pd.DataFrame(returns_data, index=data_copy.index, columns=returns_df_cols)
        trades_df = pd.DataFrame(trades_list)
        
        # Ensure correct dtypes for trades_df
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            numeric_cols = [
                'entry_price', 'quantity', 'entry_fee_frac_value', 'entry_fee_curr_value', 
                'entry_fee_total_currency', 'exit_price', 'bars_held', 'exit_fee_frac_value', 
                'exit_fee_curr_value', 'fee_total_currency', 'fee_total_as_fraction_of_equity',
                'pnl_gross_currency', 'pnl_gross_frac', 'pnl_net_currency', 'pnl_net_frac'
            ]
            for col in numeric_cols:
                if col in trades_df.columns:
                     trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')

        # Return the result object, passing the ORIGINAL data (not data_copy with temp cols)
        return BacktestResult(trades=trades_df, returns=returns_df, initial_capital=capital, final_equity=current_equity, ohlcv_data=data) # Pass original data


def backtest(
    data: pd.DataFrame,
    entry_column: str,
    exit_column: str,
    entry_price_col: str = 'close',
    exit_price_col: str = 'close',
    size: any = None, # Renamed, default None
    size_unit: str = 'fraction', # 'fraction', 'quantity', 'nominal'
    fee_frac: float = 0.0,
    fee_curr: float = 0.0,
    capital: float = 100_000.0,
    allow_fractional_positions: bool = True,
    signal_price_col: str = 'close'
) -> BacktestResult:
    """
    Functional wrapper for Backtester().backtest method.

    Args:
        data (pd.DataFrame): Input OHLCV data with a DatetimeIndex.
        entry_column (str): Column name or pandas expression for entry signals (boolean).
        exit_column (str): Column name or pandas expression for exit signals (boolean).
        entry_price_col (str, optional): Column for actual entry prices. Defaults to 'close'.
        exit_price_col (str, optional): Column for actual exit prices. Defaults to 'close'.
        size (any, optional): Size of the position.
            - If `size_unit` is 'fraction' (default):
                - If `None` (default), uses 1.0 (100% of current equity).
                - If float, specifies fraction of current equity (e.g., 0.5 for 50%).
                - If str (column name), column contains the fraction of equity.
            - If `size_unit` is 'quantity':
                - If float/int, specifies the number of units/shares.
                - If str (column name), column contains the number of units/shares.
            - If `size_unit` is 'nominal':
                - If float/int, specifies the nominal monetary value to invest.
                - If str (column name), column contains the nominal monetary value.
        size_unit (str, optional): Unit for `size`. Can be 'fraction', 'quantity', or 'nominal'.
            Defaults to 'fraction'.
        fee_frac (float, optional): Fractional fee per trade (e.g., 0.001 for 0.1%). Defaults to 0.0.
        fee_curr (float, optional): Currency fee per unit traded. Defaults to 0.0.
        capital (float, optional): Initial capital. Defaults to 100,000.0.
        allow_fractional_positions (bool, optional): Whether to allow fractional shares/units.
            Defaults to True.
        signal_price_col (str, optional): Column for mark-to-market calculations. Defaults to 'close'.

    Returns:
        BacktestResult: Object containing trades, returns, and other results.
    """
    # Create an instance of Backtester and call its backtest method
    instance = Backtester()
    return instance.backtest(
        data=data,
        entry_column=entry_column,
        exit_column=exit_column,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        size=size,
        size_unit=size_unit,
        fee_frac=fee_frac,
        fee_curr=fee_curr,
        capital=capital,
        allow_fractional_positions=allow_fractional_positions,
        signal_price_col=signal_price_col
    ) 