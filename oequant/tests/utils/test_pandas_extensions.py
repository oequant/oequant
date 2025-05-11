import pandas as pd
import numpy as np
import pytest
from unittest import mock
from unittest.mock import patch, MagicMock, ANY

# Import oequant to trigger the __init__.py and thus pandas_extensions.init_pandas_extensions()
import oequant 
# Explicitly import to check its state if needed, or call init again (it should be idempotent)
from oequant.utils import pandas_extensions 

# Fixture to ensure pandas extensions are initialized before tests in this module
@pytest.fixture(autouse=True)
def ensure_extensions_initialized():
    try:
        import oequant.utils.pandas_extensions
        oequant.utils.pandas_extensions.init_pandas_extensions()
    except ImportError as e:
        pytest.skip(f"Skipping pandas extensions tests, oequant or dependencies not fully available: {e}")

@pytest.fixture
def sample_df():
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = {
        'A': np.random.rand(10) * 100,
        'B': np.random.randn(10),
        'C': range(10)
    }
    df = pd.DataFrame(data, index=pd.Series(dates, name="Date"))
    return df

@pytest.fixture
def sample_series(sample_df):
    return sample_df['A']

def test_extensions_applied():
    "Test that methods are actually attached to DataFrame and Series."    
    assert hasattr(pd.DataFrame, 'iplot')
    assert hasattr(pd.DataFrame, 'iplot2')
    assert hasattr(pd.Series, 'iplot')
    assert hasattr(pd.Series, 'iplot2')
    assert hasattr(pd.DataFrame, 'display')
    assert hasattr(pd.DataFrame, 'ggplot')
    assert hasattr(pd.Series, 'ggplot')
    assert hasattr(pd.DataFrame, 'append') # Custom append
    assert callable(pd.qcut) # Check if pd.qcut is still callable (it's replaced)

def test_custom_qcut():
    "Test the custom pd.qcut wrapper for duplicates and inf handling." 
    s_inf = pd.Series([1, 2, 3, 4, 5, 6, np.inf, -np.inf, 7, 8])
    s_clean = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Test inf handling (should not error and infs should be NaN before qcut)
    bins_inf = pd.qcut(s_inf, 4, labels=False)
    assert bins_inf.isnull().sum() == 2 # two infs became NaNs

    # Test duplicates='drop' is default
    s_dups = pd.Series([1, 1, 1, 1, 2, 3, 4, 5, 6, 7])
    try:
        bins_dups = pd.qcut(s_dups, 4, labels=False)
        assert bins_dups.nunique() <= 4 
    except ValueError as e:
        pytest.fail(f"Custom pd.qcut should handle duplicates by default: {e}")

    bins_clean = pd.qcut(s_clean, 2, labels=False)
    assert (bins_clean == 0).sum() == 5
    assert (bins_clean == 1).sum() == 5

@patch('oequant.utils.pandas_extensions.pio') 
@patch('plotly.express.line')
@patch('plotly.graph_objects.Figure') 
def test_iplot_plotly_basic(mock_go_figure, mock_px_line, mock_pio, sample_df, sample_series):
    mock_fig_instance = MagicMock()
    mock_px_line.return_value = mock_fig_instance 
    mock_go_figure.return_value = mock_fig_instance 
    
    # Test with DataFrame
    returned_fig_df = sample_df.iplot() 
    mock_px_line.assert_called_with(ANY, template='plotly_dark')
    mock_fig_instance.update_traces.assert_called_with(line=dict(width=1.0))
    assert returned_fig_df is mock_fig_instance
    mock_px_line.reset_mock()
    mock_fig_instance.reset_mock()

    # Test with Series. _iplot converts Series to DataFrame then calls itself or plotting func.
    sample_series.iplot(kind='bar', color='red')
    mock_fig_instance.reset_mock()
    with patch('plotly.express.bar', return_value=mock_fig_instance) as mock_px_bar:
        returned_fig_series_bar = sample_series.iplot(kind='bar')
        mock_px_bar.assert_called_once()
        assert returned_fig_series_bar is mock_fig_instance

@patch('bokeh.io.output_notebook')
@patch('bokeh.io.show')
@patch('bokeh.plotting.figure') 
def test_iplot_bokeh_basic(mock_bplt_figure, mock_bokeh_io_show, mock_bokeh_io_output_notebook, sample_df):
    mock_fig_instance = MagicMock()
    mock_bplt_figure.return_value = mock_fig_instance
    mock_fig_instance.line = MagicMock()
    mock_fig_instance.add_tools = MagicMock()
    mock_fig_instance.legend = MagicMock()

    returned_fig_df = sample_df.iplot(backend='bokeh')
    mock_bokeh_io_output_notebook.assert_called_once()
    mock_bplt_figure.assert_called()
    assert mock_fig_instance.line.call_count > 0 
    mock_bokeh_io_show.assert_called_with(mock_fig_instance)
    assert returned_fig_df is mock_fig_instance

@patch.object(pd.DataFrame, 'hvplot', create=True, new_callable=MagicMock)
@patch('holoviews.extension')
def test_iplot_hvplot_basic(mock_df_hvplot_method, mock_holoviews_extension, sample_df, sample_series):
    """Test df.iplot() with hvplot backend."""
    try:
        import hvplot.pandas  # Ensure hvplot can be imported
        import hvplot.hvplot_extension
        
        # Save the original and patch directly - this is safer than mocking
        original_compatibility = getattr(hvplot.hvplot_extension, 'compatibility', None)
        try:
            # Directly modify the real module
            hvplot.hvplot_extension.compatibility = None
            
            # mock_holoviews_extension is for the call hv.extension('bokeh') in _iplot
            mock_holoviews_extension.return_value = MagicMock()
            
            mock_hvplot_instance_to_return = MagicMock()  # Renamed for clarity
            mock_df_hvplot_method.return_value = mock_hvplot_instance_to_return
            
            # Test DataFrame
            returned_fig = sample_df.iplot(backend='hvplot')
            mock_holoviews_extension.assert_called_with('bokeh')
            mock_df_hvplot_method.assert_called_once_with(kind='line', show_legend=True, width=None, height=None, color=None, subplots=False, secondary_y=None)
            assert returned_fig is mock_hvplot_instance_to_return
            
            mock_df_hvplot_method.reset_mock()
            mock_holoviews_extension.reset_mock()
            
            # Test Series (hvplot should also patch Series)
            with patch.object(pd.Series, 'hvplot', create=True, new_callable=MagicMock) as mock_series_hvplot_method:
                mock_series_hvplot_method.return_value = mock_hvplot_instance_to_return
                returned_fig_series = sample_series.iplot(backend='hvplot')
                mock_holoviews_extension.assert_called_with('bokeh')  # hv.extension is called again
                mock_series_hvplot_method.assert_called_once_with(kind='line', show_legend=True, width=None, height=None, color=None, subplots=False, secondary_y=None)
                assert returned_fig_series is mock_hvplot_instance_to_return
        finally:
            # Restore original value
            hvplot.hvplot_extension.compatibility = original_compatibility
    except ImportError:
        pytest.skip("hvplot not installed, skipping hvplot backend test for iplot.")

@patch('oequant.utils.pandas_extensions.sns')
def test_dataframe_display(mock_sns, sample_df):
    mock_cmap = MagicMock()
    mock_sns.light_palette.return_value = mock_cmap
    mock_sns.dark_palette.return_value = mock_cmap

    styler = sample_df.display(precision=3, theme='dark', background_gradient=True)
    assert isinstance(styler, pd.io.formats.style.Styler)
    mock_sns.dark_palette.assert_called_with("#69d", reverse=False, as_cmap=True)

@patch('plotnine.ggplot')
@patch('plotnine.aes')
@patch('plotnine.geom_line')
@patch('plotnine.theme_minimal')
@patch('plotnine.theme')
def test_ggplot_df_basic(mock_p9_theme, mock_p9_theme_minimal, mock_p9_geom_line, mock_p9_aes, mock_p9_ggplot, sample_df, sample_series):
    mock_plot_instance = MagicMock()
    mock_p9_ggplot.return_value = mock_plot_instance
    # Configure mock_plot_instance so that chained '+' operations return the same instance
    mock_plot_instance.__add__ = MagicMock(return_value=mock_plot_instance)
    mock_plot_instance.__iadd__ = MagicMock(return_value=mock_plot_instance) # Added for '+='

    try:
        import plotnine # Ensure plotnine can be imported
        # Test with DataFrame
        plot_df = sample_df.ggplot()
        mock_p9_ggplot.assert_called()
        assert mock_p9_geom_line.call_count > 0 
        assert plot_df is mock_plot_instance
        mock_p9_ggplot.reset_mock()
        mock_p9_geom_line.reset_mock()
        mock_plot_instance.reset_mock() # Reset for series test

        # Test with Series
        plot_series = sample_series.ggplot(geom='point')
        mock_p9_ggplot.assert_called()
        assert plot_series is mock_plot_instance
    except ImportError:
        pytest.skip("plotnine not installed, skipping ggplot test.")

def test_custom_append(sample_df):
    df_to_append = pd.DataFrame({'A': [101, 102], 'B': [0.5, -0.5], 'C': [10,11]}, index=pd.to_datetime(['2023-01-11', '2023-01-12']))
    series_to_append = pd.Series({'A': 103, 'B': 0.1, 'C': 12, 'D': 99}, name=pd.to_datetime('2023-01-13'))
    original_len = len(sample_df)

    df_app_df = sample_df.append(df_to_append)
    assert len(df_app_df) == original_len + len(df_to_append)
    assert 'D' not in df_app_df.columns

    df_app_series = sample_df.append(series_to_append, ignore_index=False) 
    assert len(df_app_series) == original_len + 1
    assert 'D' in df_app_series.columns
    assert df_app_series.loc[series_to_append.name, 'D'] == 99

    df_list_to_append = [df_to_append, df_to_append.rename(index=lambda x: x + pd.Timedelta(days=2))]
    df_app_list = sample_df.append(df_list_to_append)
    assert len(df_app_list) == original_len + sum(len(d) for d in df_list_to_append) 