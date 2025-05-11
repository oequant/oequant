import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from oequant.research.signal.quantile_analysis import research_signal_bins, _calculate_group_stats
import plotly.graph_objects as go

@pytest.fixture
def sample_df_for_quantile_analysis():
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = {
        'signal1': np.random.rand(100) * 100,
        'signal2': np.random.randn(100),
        'forward_return_01': np.random.randn(100) * 0.01, # Daily returns
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    # Add some NaNs to test handling
    df.loc[df.index[5:10], 'signal1'] = np.nan
    df.loc[df.index[10:15], 'forward_return_01'] = np.nan
    return df

# Test _calculate_group_stats separately for robustness
def test_calculate_group_stats_empty_series():
    stats = _calculate_group_stats(pd.Series([], dtype=float))
    assert stats['Count'] == 0
    assert stats['Sharpe Ratio'] == 0

def test_calculate_group_stats_basic():
    s = pd.Series([0.01, 0.02, -0.01, 0.005, 0.015])
    stats = _calculate_group_stats(s)
    assert stats['Count'] == 5
    assert np.isclose(stats['Return mean %'], s.mean() * 100)
    assert stats['Win Rate'] == 0.8
    assert stats['Max DD %'] < 0 # Should be negative

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False) # Assume not in notebook for tests
@patch('oequant.research.signal.quantile_analysis.display') # Mock display
@patch('plotly.express.line') # Mock plotly express line
def test_research_signal_bins_qcut_basic(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis
    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig

    results = research_signal_bins(
        df.copy(), 
        signal_cols='signal1', 
        forward_ret_col='forward_return_01', 
        split_kind='qcut', 
        split_params=5,
        show_stats=True,
        show_pnl_plot=True
    )
    
    assert isinstance(results, dict)
    assert 'stats_df' in results
    assert 'pnl_fig' in results
    assert isinstance(results['stats_df'], pd.DataFrame)
    assert not results['stats_df'].empty
    assert results['pnl_fig'] is mock_fig
    assert 'signal1_bin' in results['stats_df'].index.names
    mock_px_line.assert_called_once()
    # Check if tabulate was called via display(print(...)) or print directly for stats
    # This depends on how the mocked display behaves vs print within the function

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_cut_custom_bins(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis
    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig

    custom_bins = [0, 20, 40, 60, 80, 100]
    results = research_signal_bins(
        df.copy(), 
        signal_cols='signal1', 
        forward_ret_col='forward_return_01', 
        split_kind='cut', 
        split_params=custom_bins
    )
    assert not results['stats_df'].empty
    assert results['stats_df'].shape[0] <= len(custom_bins) -1 # Number of bins
    mock_px_line.assert_called_once()

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_oos_separated(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.copy()
    # Ensure we have enough non-NaN data
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig
    
    oos_date = df.index[50]  # Split at the middle of the dataset
    
    # Call function with show_oos_separated=True
    results = research_signal_bins(
        df, 
        signal_cols='signal1',
        forward_ret_col='forward_return_01',
        split_params=3,
        oos_from=oos_date,
        show_oos_separated=True
    )
    
    # Verify structure of results
    assert 'in_sample' in results
    assert 'out_sample' in results
    assert 'stats_df' in results['in_sample']
    assert 'stats_df' in results['out_sample']
    assert 'pnl_fig' in results['in_sample']
    assert 'pnl_fig' in results['out_sample']
    
    # Verify in-sample data is before oos_date and out-sample is after
    assert not results['in_sample']['stats_df'].empty
    assert not results['out_sample']['stats_df'].empty
    
    # Check that 'is_oos' is not in the index levels
    assert 'is_oos' not in results['in_sample']['stats_df'].index.names
    assert 'is_oos' not in results['out_sample']['stats_df'].index.names
    
    # Both should have signal1_bin
    assert 'signal1_bin' in results['in_sample']['stats_df'].index.names
    assert 'signal1_bin' in results['out_sample']['stats_df'].index.names
    
    # Verify plots were created
    assert mock_px_line.call_count == 2
    assert results['in_sample']['pnl_fig'] == mock_fig
    assert results['out_sample']['pnl_fig'] == mock_fig

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_oos(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis
    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig
    
    oos_date = df.index[50]
    results = research_signal_bins(
        df.copy(), 
        signal_cols='signal2', 
        forward_ret_col='forward_return_01', 
        split_params=3,
        oos_from=oos_date
    )
    assert not results['stats_df'].empty
    assert 'is_oos' in results['stats_df'].index.names
    assert 'signal2_bin' in results['stats_df'].index.names
    assert len(results['stats_df'].index.levels[0]) == 2 # True/False for is_oos
    mock_px_line.assert_called_once()

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_multiple_signals(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.copy()
    # Ensure signal2 doesn't have too many NaNs that would make binning impossible for small q
    df['signal2'] = df['signal2'].fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(subset=['signal1', 'signal2', 'forward_return_01'])

    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig
    
    results = research_signal_bins(
        df, 
        signal_cols=['signal1', 'signal2'], 
        forward_ret_col='forward_return_01', 
        split_params=2 # Small number of quantiles to avoid issues with limited data after NaN drop
    )
    assert not results['stats_df'].empty
    assert 'signal1_bin' in results['stats_df'].index.names
    assert 'signal2_bin' in results['stats_df'].index.names
    mock_px_line.assert_called_once()

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_no_plot_no_stats(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis
    results = research_signal_bins(
        df.copy(), 
        signal_cols='signal1', 
        forward_ret_col='forward_return_01', 
        show_stats=False, 
        show_pnl_plot=False
    )
    assert not results['stats_df'].empty # Stats are still calculated
    assert results['pnl_fig'] is None # Figure object is None
    mock_px_line.assert_not_called() # Plotting function not called
    # mock_display might be called for warnings/errors, so not asserting not_called here without more specific mock


@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_date_col_param(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.reset_index()
    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig

    results = research_signal_bins(
        df.copy(),
        signal_cols='signal1',
        forward_ret_col='forward_return_01',
        split_kind='qcut',
        split_params=5,
        date_col='Date' # Use the new 'Date' column
    )
    assert not results['stats_df'].empty
    assert 'signal1_bin' in results['stats_df'].index.names
    mock_px_line.assert_called_once()

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_all_nan_signal(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.copy()
    df['signal1'] = np.nan # Make one signal all NaNs
    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig

    results = research_signal_bins(
        df,
        signal_cols=['signal1', 'signal2'],
        forward_ret_col='forward_return_01',
        split_params=3
    )
    # signal1 should be skipped, but signal2 should still be processed
    assert 'signal2_bin' in results['stats_df'].index.names
    assert 'signal1_bin' not in results['stats_df'].index.names # Because it was skipped
    assert not results['stats_df'].empty
    mock_px_line.assert_called_once() # Plot based on signal2

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_empty_after_dropna(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.copy()
    df['forward_return_01'] = np.nan # Make all forward returns NaN

    results = research_signal_bins(
        df,
        signal_cols='signal1',
        forward_ret_col='forward_return_01',
        split_params=3
    )
    assert results['stats_df'].empty
    assert results['pnl_fig'] is None
    mock_px_line.assert_not_called()

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_multiple_signals_with_nan(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.copy()
    # Ensure signal2 doesn't have too many NaNs that would make binning impossible for small q
    df['signal2'] = df['signal2'].fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(subset=['signal1', 'signal2', 'forward_return_01'])

    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig
    
    results = research_signal_bins(
        df, 
        signal_cols=['signal1', 'signal2'], 
        forward_ret_col='forward_return_01', 
        split_params=2 # Small number of quantiles to avoid issues with limited data after NaN drop
    )
    assert not results['stats_df'].empty
    assert 'signal1_bin' in results['stats_df'].index.names
    assert 'signal2_bin' in results['stats_df'].index.names
    mock_px_line.assert_called_once()

@patch('oequant.research.signal.quantile_analysis._is_notebook', return_value=False)
@patch('oequant.research.signal.quantile_analysis.display')
@patch('plotly.express.line')
def test_research_signal_bins_multiple_signals_with_nan_and_oos(mock_px_line, mock_display, mock_is_notebook, sample_df_for_quantile_analysis):
    df = sample_df_for_quantile_analysis.copy()
    # Ensure signal2 doesn't have too many NaNs that would make binning impossible for small q
    df['signal2'] = df['signal2'].fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(subset=['signal1', 'signal2', 'forward_return_01'])

    mock_fig = MagicMock(spec=go.Figure)
    mock_px_line.return_value = mock_fig
    
    oos_date = df.index[50]
    results = research_signal_bins(
        df, 
        signal_cols=['signal1', 'signal2'], 
        forward_ret_col='forward_return_01', 
        split_params=2,
        oos_from=oos_date
    )
    assert not results['stats_df'].empty
    assert 'signal1_bin' in results['stats_df'].index.names
    assert 'signal2_bin' in results['stats_df'].index.names
    assert 'is_oos' in results['stats_df'].index.names
    mock_px_line.assert_called_once() 