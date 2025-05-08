import pandas as pd
import numpy as np
from oequant.backtesting.results import BacktestResult # Adjusted import path

def _get_annualization_factor(returns_index: pd.DatetimeIndex) -> float:
    if len(returns_index) < 2:
        return 252 # Default if not enough data to infer
    median_diff = (returns_index[1:] - returns_index[:-1]).median()
    if median_diff <= pd.Timedelta(days=1):
        return 252 # Daily
    elif median_diff <= pd.Timedelta(days=7):
        return 52  # Weekly
    elif median_diff <= pd.Timedelta(days=31):
        return 12  # Monthly
    return 1     # Assume already annualized or unknown

def calculate_statistics(result: BacktestResult, PnL_type: str = 'net', risk_free_rate_annual: float = 0.0) -> pd.Series:
    """
    Calculates various performance statistics from a BacktestResult.

    Args:
        result (BacktestResult): The result object from a backtest.
        PnL_type (str, optional): Specifies whether to use 'gross' or 'net' P&L for calculations.
                                  Defaults to 'net'.
        risk_free_rate_annual (float, optional): Annual risk-free rate for Sharpe/Sortino. Defaults to 0.0.

    Returns:
        pd.Series: A pandas Series containing the calculated performance statistics.
    """
    if PnL_type not in ['gross', 'net']:
        raise ValueError("PnL_type must be either 'gross' or 'net'.")

    stats = {}
    trades_df = result.trades
    returns_df = result.returns.copy() # Work on a copy

    if returns_df.empty:
        # Return default/NaN stats if no returns data
        stats['cagr_pct'] = 0.0
        stats['return_per_trade_pct'] = 0.0
        stats['sharpe_ratio'] = np.nan
        stats['sortino_ratio'] = np.nan
        stats['serenity_ratio'] = np.nan
        stats['pct_in_position'] = 0.0
        stats['max_dd_pct'] = 0.0
        stats['cagr_to_max_dd'] = np.nan
        stats['total_trades'] = 0
        stats['win_rate_pct'] = 0.0
        stats['loss_rate_pct'] = 0.0
        stats['avg_win_trade_pct'] = 0.0
        stats['avg_loss_trade_pct'] = 0.0
        stats['profit_factor'] = np.nan
        stats['avg_bars_held'] = 0.0
        return pd.Series(stats)

    # Determine PnL columns to use
    ret_col_frac = f'return_{PnL_type}_frac'
    trade_pnl_col_frac = f'pnl_{PnL_type}_frac'
    trade_pnl_col_curr = f'pnl_{PnL_type}_currency'

    # CAGR
    initial_equity = result.initial_capital
    final_equity = result.final_equity
    num_periods = len(returns_df)
    annualization_factor = _get_annualization_factor(returns_df.index)
    
    if num_periods > 0 and annualization_factor > 0:
        total_return = (final_equity / initial_equity) - 1
        years = num_periods / annualization_factor
        if years >= 1.0: # Annualize if 1 year or more
            stats['cagr_pct'] = ((1 + total_return) ** (1 / years) - 1) * 100 if total_return > -1 else -100.0
        elif years > 0: # For periods less than a year, show non-annualized total return
            stats['cagr_pct'] = total_return * 100
        else: # Should not happen if num_periods > 0
            stats['cagr_pct'] = 0.0
    else:
        stats['cagr_pct'] = 0.0

    # Return per trade
    if not trades_df.empty and trade_pnl_col_frac in trades_df.columns:
        stats['return_per_trade_pct'] = trades_df[trade_pnl_col_frac].mean() * 100
    else:
        stats['return_per_trade_pct'] = 0.0

    # Sharpe Ratio & Sortino Ratio
    period_returns = returns_df[ret_col_frac].dropna()
    risk_free_rate_period = (1 + risk_free_rate_annual)**(1/annualization_factor) - 1 if annualization_factor > 0 else risk_free_rate_annual
    
    if len(period_returns) > 1:
        mean_excess_return = period_returns.mean() - risk_free_rate_period
        std_return = period_returns.std()
        if std_return != 0 and not np.isnan(std_return):
            stats['sharpe_ratio'] = (mean_excess_return / std_return) * np.sqrt(annualization_factor)
        else:
            stats['sharpe_ratio'] = np.nan

        negative_returns = period_returns[period_returns < risk_free_rate_period] - risk_free_rate_period
        downside_std = negative_returns.std()
        if downside_std != 0 and not np.isnan(downside_std):
            stats['sortino_ratio'] = (mean_excess_return / downside_std) * np.sqrt(annualization_factor)
        else:
            stats['sortino_ratio'] = np.nan if stats['sharpe_ratio'] is np.nan else (np.inf if mean_excess_return > 0 else (-np.inf if mean_excess_return <0 else 0))

    else:
        stats['sharpe_ratio'] = np.nan
        stats['sortino_ratio'] = np.nan

    # Pct in position
    stats['pct_in_position'] = (returns_df['position_unit'] != 0).sum() / num_periods * 100 if num_periods > 0 else 0.0

    # Max Drawdown
    equity_curve = returns_df['equity']
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    stats['max_dd_pct'] = drawdown.min() * 100 if not drawdown.empty else 0.0
    
    # Absolute Max Drawdown (for Serenity)
    peak_before_max_dd = rolling_max[drawdown.idxmin()] if not drawdown.empty and drawdown.min() < 0 else initial_equity
    max_abs_dd_currency = abs(stats['max_dd_pct']/100 * peak_before_max_dd) if peak_before_max_dd > 0 else 0

    # CAGR to Max DD
    cagr_val = stats['cagr_pct']
    # max_dd_pct is already in percentage and usually negative or zero.
    # We need its absolute magnitude for the ratio.
    max_dd_magnitude = abs(stats['max_dd_pct'])

    if max_dd_magnitude != 0:
        stats['cagr_to_max_dd'] = cagr_val / max_dd_magnitude
    else: # max_dd_magnitude is 0
        if cagr_val > 0:
            stats['cagr_to_max_dd'] = np.inf
        elif cagr_val < 0:
            stats['cagr_to_max_dd'] = -np.inf
        else: # Both CAGR and MaxDD are 0 (cagr_val is 0)
            stats['cagr_to_max_dd'] = np.nan

    # Serenity Ratio (Total Net Profit / Max Absolute Drawdown Currency)
    total_net_profit_currency = final_equity - initial_equity
    if max_abs_dd_currency != 0:
        stats['serenity_ratio'] = total_net_profit_currency / max_abs_dd_currency
    else: # max_abs_dd_currency is 0
        if total_net_profit_currency == 0:
            stats['serenity_ratio'] = np.nan # Align with test expectation for no trades, or 0 trades with 0 profit
        else: # Has profit but no drawdown
            stats['serenity_ratio'] = np.inf
        
    # Additional trade statistics
    stats['total_trades'] = len(trades_df)
    if not trades_df.empty and trade_pnl_col_curr in trades_df.columns:
        winning_trades = trades_df[trades_df[trade_pnl_col_curr] > 0]
        losing_trades = trades_df[trades_df[trade_pnl_col_curr] < 0]
        
        stats['win_rate_pct'] = (len(winning_trades) / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0.0
        stats['loss_rate_pct'] = (len(losing_trades) / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0.0

        if trade_pnl_col_frac in trades_df.columns:
            stats['avg_win_trade_pct'] = winning_trades[trade_pnl_col_frac].mean() * 100 if not winning_trades.empty else 0.0
            stats['avg_loss_trade_pct'] = losing_trades[trade_pnl_col_frac].mean() * 100 if not losing_trades.empty else 0.0
        else:
            stats['avg_win_trade_pct'] = np.nan
            stats['avg_loss_trade_pct'] = np.nan
            
        sum_gains = winning_trades[trade_pnl_col_curr].sum()
        sum_losses = abs(losing_trades[trade_pnl_col_curr].sum())
        stats['profit_factor'] = sum_gains / sum_losses if sum_losses != 0 else np.inf
        if sum_gains == 0 and sum_losses == 0: stats['profit_factor'] = 1.0 # No PnL no loss -> factor 1

    else:
        stats['win_rate_pct'] = 0.0
        stats['loss_rate_pct'] = 0.0
        stats['avg_win_trade_pct'] = 0.0
        stats['avg_loss_trade_pct'] = 0.0
        stats['profit_factor'] = 1.0 # Align with test for no trades

    # Average Bars Held
    if not trades_df.empty and 'bars_held' in trades_df.columns:
        stats['avg_bars_held'] = trades_df['bars_held'].mean()
    else:
        stats['avg_bars_held'] = 0.0

    # Round floats before converting to Series, but keep NaN/inf as is
    for k, v in stats.items():
        if isinstance(v, float) and not (np.isnan(v) or np.isinf(v)):
            stats[k] = round(v, 4)
            
    return pd.Series(stats) 