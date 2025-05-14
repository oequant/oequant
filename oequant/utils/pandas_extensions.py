import pandas as pd
import seaborn as sns
import numpy as np
import plotly.io as pio
pio.renderers.default='vscode' # 'notebook' or 'browser', 'vscode', 'kaggle', 'iframe'

"""

Usage:
import oequant.utils.pandas_extensions as pd_ext # or simply import oequant if init is in top __init__
# pd_ext.init_pandas_extensions() # Call this if not initialized by oequant top-level import

# df['DE_SS'].ggplot()

"""

# Module-level flag to ensure initialization runs once
_INITIALIZED = False

def _style_bokeh_axes(p, x_is_datetime=False):
    """
    Apply consistent styling to Bokeh axes for better visibility.
    
    Args:
        p: Bokeh figure object
        x_is_datetime: Boolean flag for whether x-axis contains datetime values
    """
    from bokeh.models import DatetimeTickFormatter, DatetimeTicker, HoverTool
    
    p.xaxis.axis_label_text_color = "white"
    p.yaxis.axis_label_text_color = "white"
    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"
    p.xaxis.major_tick_line_color = "white"
    p.yaxis.major_tick_line_color = "white"
    p.xaxis.minor_tick_line_color = "white"
    p.yaxis.minor_tick_line_color = "white"
    p.xaxis.axis_line_color = "white"
    p.yaxis.axis_line_color = "white"
    p.xaxis.major_label_text_font_size = "9pt"
    p.yaxis.major_label_text_font_size = "9pt"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.axis_label_text_font_size = "10pt"
    p.xaxis.major_tick_line_width = 1.5
    p.yaxis.major_tick_line_width = 1.5
    p.xaxis.axis_line_width = 1.5
    p.yaxis.axis_line_width = 1.5
    p.xaxis.major_tick_out = 5
    p.yaxis.major_tick_out = 5
    p.xaxis.axis_label = "Date" if x_is_datetime else "X"
    p.yaxis.axis_label = "Value"
    
    if x_is_datetime:
        p.xaxis.formatter = DatetimeTickFormatter(
            milliseconds="%H:%M:%S.%3N",
            seconds="%H:%M:%S",
            minsec="%H:%M:%S",
            minutes="%H:%M",
            hourmin="%H:%M",
            hours="%H:%M",
            days="%Y-%m-%d",
            months="%Y-%m",
            years="%Y"
        )
        p.xaxis.ticker = DatetimeTicker()
    
    p.outline_line_color = "white"
    p.outline_line_width = 1.5
    
    if p.legend:
        p.legend.visible = True
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_color = "#1a1a1a"
        p.legend.background_fill_alpha = 0.3
        p.legend.label_text_color = "white"
        p.legend.label_text_font_size = "8pt"
        p.legend.border_line_color = "white"
        p.legend.border_line_alpha = 0.2
        p.legend.padding = 5
        p.legend.spacing = 3
        p.legend.margin = 2
        p.legend.label_height = 10
        p.legend.label_text_alpha = 0.8
    
    hover_tool = None
    for tool in p.tools:
        if isinstance(tool, HoverTool):
            hover_tool = tool
            break
    
    if hover_tool is None:
        hover_tool = HoverTool()
        p.add_tools(hover_tool)
    
    if x_is_datetime:
        hover_tool.tooltips = [
            ('Date', '@x{%F}'),
            ('Time', '@x{%H:%M:%S}'),
            ('Value', '@y{0.0000}')
        ]
        hover_tool.formatters = {'@x': 'datetime'}
    else:
        hover_tool.tooltips = [
            ('X', '@x'),
            ('Value', '@y{0.0000}')
        ]
    
    hover_tool.mode = 'mouse'
    hover_tool.point_policy = 'snap_to_data'
    hover_tool.line_policy = 'nearest'
    p.grid.grid_line_color = "#5e5e5e"
    p.grid.grid_line_alpha = 0.2
    p.xgrid.grid_line_color = "#5e5e5e"
    p.ygrid.grid_line_color = "#5e5e5e"
    p.xgrid.grid_line_alpha = 0.2
    p.ygrid.grid_line_alpha = 0.2
    p.background_fill_color = "#1a1a1a"
    p.border_fill_color = "#1a1a1a"

def _iplot(self, backend='bokeh', kind='line', use_webgl=False, **kwargs):
    # df = self.copy() # Original line
    # Handle Series input by converting to DataFrame
    if isinstance(self, pd.Series):
        df = self.to_frame(name=self.name or 'value')
    else:
        df = self.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in df.columns]

    legend = kwargs.pop('legend', True)
    width = kwargs.pop('width', None)
    height = kwargs.pop('height', None)
    color = kwargs.pop('color', None)
    dimensions = kwargs.pop('dimensions', None)
    figsize = kwargs.pop('figsize', None)
    secondary_y = kwargs.pop('secondary_y', None)
    subplots = kwargs.pop('subplots', False)
    legend_position = kwargs.pop('legend_position', 'internal')

    if dimensions:
        width, height = dimensions
    elif figsize:
        width, height = [int(x * 100) for x in figsize]

    x_is_datetime = isinstance(df.index, pd.DatetimeIndex)
    x_data = df.index
    
    if 'x' in kwargs and kwargs['x'] in df.columns:
        x_data = df[kwargs['x']]
        x_is_datetime = isinstance(x_data, pd.Series) and pd.api.types.is_datetime64_any_dtype(x_data)

    if backend == 'hvplot':
        try:
            import hvplot.pandas
            import holoviews as hv
            hv.extension('bokeh')
            # hv.renderer('bokeh').theme = 'dark' # This might need to be configured elsewhere or removed if problematic
            kwargs['show_legend'] = legend
            if width:
                kwargs['width'] = width
            if height:
                kwargs['height'] = height
            if color:
                kwargs['color'] = color
            if subplots:
                kwargs['subplots'] = True
            if secondary_y:
                kwargs['secondary_y'] = secondary_y
            plot = self.hvplot(kind=kind, **kwargs)
            # hv.render(plot, backend='bokeh').output_backend = 'webgl' # Rendering here might be too soon
            return plot
        except ImportError:
            # print("hvplot not installed, skipping hvplot backend.")
            raise ImportError("Install hvplot: 'pip install hvplot'")
    elif backend == 'bokeh':
        from bokeh.models import Range1d, LinearAxis, DatetimeTickFormatter, Legend
        import bokeh.plotting as bplt
        from bokeh.io import output_notebook, show
        
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ]
        
        try:
            output_notebook(hide_banner=True)
        except Exception: # Broad exception for environments where this might fail
            pass # print("Could not initialize Bokeh notebook mode.")
        
        df_plot = self.replace([np.inf, -np.inf], np.nan) # Bokeh doesn't handle inf well for auto-ranging
        # For fillna(0), it might be better to let user decide or handle it per plot type

        width = kwargs.get('width', 1200)
        height = kwargs.get('height', 600)
        plot_width = width
        if legend_position == 'external':
            plot_width = int(width * 0.8)
        
        tools = "pan,wheel_zoom,box_zoom,reset,save,hover"
        output_backend_val = "webgl" if use_webgl else "canvas"
        
        figure_args = {
            'width': plot_width, 
            'height': height, 
            'tools': tools,
            'toolbar_location': "above",
            'output_backend': output_backend_val,
            'background_fill_color': "#1a1a1a",
            'border_fill_color': "#1a1a1a",
            'outline_line_color': "white",
            'x_axis_type': "datetime" if x_is_datetime else "auto"
        }

        if df_plot.ndim == 1 or df_plot.shape[1] == 1:
            y_data = df_plot.values if df_plot.ndim == 1 else df_plot.iloc[:, 0].values
            legend_label = getattr(df_plot, 'name', 'Value') if df_plot.ndim == 1 else str(df_plot.columns[0])
            
            p = bplt.figure(**figure_args)
            line_args = {'line_width':2, 'color':colors[0], 'alpha':0.9}
            if legend_position == 'internal' and legend:
                line_args['legend_label'] = str(legend_label)
            r = p.line(x_data, y_data, **line_args)
            _style_bokeh_axes(p, x_is_datetime)
            
            if legend and legend_position == 'external':
                from bokeh.layouts import row
                external_legend = Legend(items=[(str(legend_label), [r])], location="center", orientation="vertical")
                # Apply styling to external_legend as in _style_bokeh_axes legend part
                p_legend_panel = bplt.figure(width=int(width * 0.2), height=height, toolbar_location=None,
                                              background_fill_color="#1a1a1a", border_fill_color="#1a1a1a")
                p_legend_panel.add_layout(external_legend)
                p_legend_panel.outline_line_alpha = 0
                p_legend_panel.axis.visible = False
                layout = row(p, p_legend_panel)
                show(layout)
                return layout
            show(p)
            return p
        else:
            p = bplt.figure(**figure_args)
            renderers_legend = []
            for i, col_name in enumerate(df_plot.columns):
                color_idx = i % len(colors)
                line_args = {'line_width':2, 'color':colors[color_idx], 'alpha':0.9}
                if legend_position == 'internal' and legend:
                    line_args['legend_label'] = str(col_name)
                r = p.line(x_data, df_plot[col_name].values, **line_args)
                if legend_position == 'external' and legend: # Collect for external legend
                     renderers_legend.append((str(col_name), [r]))
            _style_bokeh_axes(p, x_is_datetime)
            
            if legend and legend_position == 'external':
                from bokeh.layouts import row
                external_legend = Legend(items=renderers_legend, location="center", orientation="vertical")
                # Apply styling to external_legend
                p_legend_panel = bplt.figure(width=int(width * 0.2), height=height, toolbar_location=None,
                                             background_fill_color="#1a1a1a", border_fill_color="#1a1a1a")
                p_legend_panel.add_layout(external_legend)
                p_legend_panel.outline_line_alpha = 0
                p_legend_panel.axis.visible = False
                layout = row(p, p_legend_panel)
                show(layout)
                return layout
            show(p)
            return p
    elif backend == 'plotly':
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            line_width_val = kwargs.pop('line_width', 1.0)
            template = kwargs.pop('template', 'plotly_dark')

            fig_list = [] # For subplots=True where px returns a list of figures

            if subplots:
                x_col_name = kwargs.get('x')
                y_cols_subplot = df.columns.tolist()
                if x_col_name and x_col_name in y_cols_subplot:
                    y_cols_subplot.remove(x_col_name)
                
                num_subplots = len(y_cols_subplot)
                if num_subplots == 0: return None # Or raise error

                fig = make_subplots(
                    rows=num_subplots, cols=1, 
                    subplot_titles=[str(c) for c in y_cols_subplot], 
                    shared_xaxes=True, vertical_spacing=0.05
                )
                for i, y_col_s in enumerate(y_cols_subplot):
                    _x = x_data if not x_col_name else df[x_col_name]
                    _y = df[y_col_s]
                    trace_type = go.Scattergl if use_webgl and kind in ['scatter', 'line'] else go.Scatter
                    trace = trace_type(
                        x=_x, y=_y, name=str(y_col_s), 
                        mode='lines' if kind == 'line' else 'markers',
                        line=dict(width=line_width_val)
                    )
                    fig.add_trace(trace, row=i+1, col=1)
                fig.update_layout(template=template, showlegend=legend, width=width, height=height, title=kwargs.get('title'))
            elif kind in px.__dict__ and not use_webgl and not secondary_y: # Basic px usage
                # This path is for px direct calls when not using webgl or secondary_y manually
                kwargs['template'] = template
                if color:
                    kwargs['color_discrete_sequence'] = [color] if isinstance(color, str) else color
                fig = getattr(px, kind)(df, **kwargs)
                if kind == 'line': # Only apply line width for line plots
                    fig.update_traces(line=dict(width=line_width_val))
            else: # Manual go.Figure construction for webgl, secondary_y, or more control
                fig = go.Figure()
                y_plot_cols = df.columns
                if 'x' in kwargs and kwargs['x'] in y_plot_cols:
                    y_plot_cols = y_plot_cols.drop(kwargs['x'])
                
                main_y_cols = [c for c in y_plot_cols if not secondary_y or c not in secondary_y]
                
                for i, col_name in enumerate(main_y_cols):
                    trace_type = go.Scattergl if use_webgl else go.Scatter
                    fig.add_trace(trace_type(
                        x=x_data if 'x' not in kwargs else df[kwargs['x']], 
                        y=df[col_name], name=str(col_name), 
                        mode='lines' if kind == 'line' else 'markers',
                        line=dict(width=line_width_val, color= (color[i % len(color)] if isinstance(color, list) else color) if color else None)
                    ))
                
                if secondary_y:
                    for i, sec_col_name in enumerate(secondary_y):
                        if sec_col_name not in df.columns: continue
                        trace_type = go.Scattergl if use_webgl else go.Scatter
                        fig.add_trace(trace_type(
                            x=x_data if 'x' not in kwargs else df[kwargs['x']],
                            y=df[sec_col_name], name=str(sec_col_name) + " (right)", yaxis="y2",
                            mode='lines' if kind == 'line' else 'markers',
                            line=dict(width=line_width_val)
                        ))
                    fig.update_layout(yaxis2=dict(title="Secondary", overlaying='y', side='right'))

                fig.update_layout(template=template, showlegend=legend, width=width, height=height, title=kwargs.get('title'))
            
            # fig.show() # Showing should be handled by the caller or notebook environment
            return fig
        except ImportError:
            # print("plotly not installed, skipping plotly backend.")
            raise ImportError("Install plotly: 'pip install plotly'")
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose from 'plotly', 'bokeh', 'hvplot'.")

from pandas import DataFrame, Series, Index as pd_Index # Renamed to avoid conflict

def append(self, other, ignore_index: bool = False, verify_integrity: bool = False, sort: bool = False) -> pd.DataFrame:
    if isinstance(other, (Series, dict)):
        if isinstance(other, dict):
            if not ignore_index:
                raise TypeError("Can only append a dict if ignore_index=True")
            other = Series(other)
        if other.name is None and not ignore_index:
            raise TypeError("Can only append a Series if ignore_index=True or if the Series has a name")
        index = pd_Index([other.name], name=self.index.name)
        idx_diff = other.index.difference(self.columns)
        combined_columns = self.columns.append(idx_diff)
        other = (other.reindex(combined_columns, copy=False).to_frame().T.infer_objects().rename_axis(index.names, copy=False))
        if not self.columns.equals(combined_columns):
            self = self.reindex(columns=combined_columns)
    elif isinstance(other, list):
        if not other: pass # Do nothing if list is empty
        elif not all(isinstance(item, DataFrame) for item in other):
             # Attempt to convert if list of dicts or similar, but be cautious
            try:
                other_df = DataFrame(other)
                # If conversion is successful and self has columns, try to align
                if not self.columns.empty and (self.columns.get_indexer(other_df.columns) >= 0).all():
                     other = [other_df.reindex(columns=self.columns)] # Make it a list of one DataFrame
                else:
                    other = [other_df] # List of one DataFrame
            except Exception:
                 raise TypeError("List elements must be DataFrames or convertible to a DataFrame compatible with self")       
    from pandas.core.reshape.concat import concat
    to_concat = [self] + (other if isinstance(other, list) else [other])
    return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort).__finalize__(self, method="append")

def dataframe_display(self, precision=2, background_gradient=True, multiline_cols=False, rotation=None, 
                        sticky_cols=True, sticky_rows=False, theme='light', *args, **kwargs):
    styled_df = self.style
    if precision is not None:
        try: # Pandas < 2.0
            styled_df = styled_df.set_precision(precision)
        except AttributeError: # Pandas >= 2.0
            styled_df = styled_df.format(precision=precision)
    
    styles_list = []
    if rotation is not None:
        styles_list.append(dict(selector="th.col_heading", props=[
            ("transform", f"rotate({rotation}deg)"), ("height", "100px"),
            ("vertical-align", "bottom"), ("padding-bottom", "10px"),
        ]))
    if styles_list:
        styled_df = styled_df.set_table_styles(styles_list)
    
    if multiline_cols:
        # Ensure columns are strings before replacing
        styled_df.data.columns = [str(x).replace(" ", "\n").replace("_", "\n") for x in styled_df.data.columns]
    
    if background_gradient:
        if 'cmap' not in kwargs:
            cmap_palette = "#69d" if theme == 'dark' else "#05a"
            cmap_func = sns.dark_palette if theme == 'dark' else sns.light_palette
            kwargs['cmap'] = cmap_func(cmap_palette, reverse=False, as_cmap=True)
        styled_df = styled_df.background_gradient(*args, **kwargs)
    
    if sticky_cols: styled_df = styled_df.set_sticky(axis=1)
    if sticky_rows: styled_df = styled_df.set_sticky(axis=0)
    return styled_df

def ggplot_df(self, x=None, y=None, spec=None, color_dict=None, alpha=0.75, size=1, geom='line', figure_size='infer', set_vscode_theme=True):
    try:
        import plotnine as p9
    except ImportError:
        # print("plotnine not installed, ggplot extension not available.")
        raise ImportError("Install plotnine: 'pip install plotnine'")

    df_to_plot = self.copy()
    if isinstance(df_to_plot, pd.Series):
        df_to_plot = df_to_plot.to_frame(name=df_to_plot.name or 'value')
    
    if isinstance(df_to_plot.index, pd.DatetimeIndex) and 'Year' not in df_to_plot.columns:
        df_to_plot['Year'] = df_to_plot.index.year
    
    if x is None:
        if isinstance(df_to_plot.index, pd.MultiIndex):
             # If MultiIndex, plotnine might need explicit x. Defaulting to reset_index and using 'index' or first level.
             # This part might need more sophisticated handling based on user intent for MultiIndex.
             print("Warning: DataFrame has MultiIndex. Resetting index for plotting. Specify 'x' explicitly for better control.")
             df_to_plot = df_to_plot.reset_index()
             x = df_to_plot.columns[0] # Use the first level of the former index
        else:
            df_to_plot['_index_x'] = df_to_plot.index # Avoid modifying original index name directly in aes
            x = '_index_x'
    
    y_cols = y if y is not None else [col for col in df_to_plot.columns if col not in [x, 'Year']]
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if not y_cols: # If y_cols is empty (e.g. only index and x were present)
        if len(df_to_plot.columns) == 1 and x == df_to_plot.columns[0]: # Plotting the index itself as y
             y_cols = [x] # This will plot x vs x, which is probably not intended but avoids error
        else: # Try to pick first numerical column if no y is specified
            num_cols = df_to_plot.select_dtypes(include=np.number).columns.tolist()
            if x in num_cols: num_cols.remove(x)
            if 'Year' in num_cols and 'Year' != x : num_cols.remove('Year')
            if num_cols: y_cols = [num_cols[0]]
            else: raise ValueError("No suitable y column found for ggplot.")

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if color_dict is None:
        color_dict = {col: default_colors[i % len(default_colors)] for i, col in enumerate(y_cols)}
    
    plot = p9.ggplot(df_to_plot, p9.aes(x=x))
    for col_y in y_cols:
        if col_y not in df_to_plot.columns: continue
        geom_args = {'color': color_dict.get(col_y, default_colors[0]), 'size': size, 'alpha': alpha}
        if geom == 'line':
            plot += p9.geom_line(p9.aes(y=col_y), **geom_args)
        elif geom == 'point':
            plot += p9.geom_point(p9.aes(y=col_y), **geom_args)
    
    plot += p9.theme_minimal()
    has_facets = False
    if spec:
        specs_list = spec if isinstance(spec, (list, tuple)) else [spec]
        for s_item in specs_list:
            plot += s_item
            if isinstance(s_item, (p9.facets.facet_wrap, p9.facets.facet_grid)):
                has_facets = True

    if set_vscode_theme:
        fig_size_val = figure_size
        if fig_size_val == 'infer':
            fig_size_val = (15, 10) if has_facets else (10, 4)
        plot += p9.theme(
            figure_size=fig_size_val,
            plot_background=p9.element_rect(fill='white'), panel_background=p9.element_rect(fill='#1e1e1e'), # VS Code dark bg
            panel_grid_major=p9.element_line(color='#404040', size=0.5),
            panel_grid_minor=p9.element_line(color='#2c2c2c', size=0.25),
            strip_background=p9.element_rect(fill='#2c2c2c'), strip_text=p9.element_text(color='#cccccc'),
            axis_text=p9.element_text(color='#cccccc'), axis_title=p9.element_text(color='#cccccc'),
            title=p9.element_text(color='#cccccc'), legend_background=p9.element_rect(fill='#1e1e1e'),
            legend_key=p9.element_rect(fill='#1e1e1e'), legend_text=p9.element_text(color='#cccccc')
        )
    return plot

_GLOBAL_INITIALIZED = False # Renamed to avoid clash if this file is imported multiple times somehow

def init_pandas_extensions():
    global _GLOBAL_INITIALIZED
    if _GLOBAL_INITIALIZED:
        return

    pd.DataFrame.iplot = _iplot
    pd.Series.iplot = _iplot # Series can also use iplot, assuming it's compatible
    pd.DataFrame.iplot2 = _iplot # Compatibility
    pd.Series.iplot2 = _iplot

    # Replace pd.qcut globally
    original_qcut = pd.qcut
    def custom_qcut(x, q, labels=None, retbins: bool = False, precision: int = 3,
                    duplicates: str = 'drop', **kwargs):
        # Ensure x is a Series for replace
        if not isinstance(x, pd.Series):
            x_series = pd.Series(x)
        else:
            x_series = x
        x_processed = x_series.replace([-np.inf, np.inf], np.nan)
        # If original was not a series, convert back if necessary, or pass series to qcut
        return original_qcut(x_processed, q, labels=labels, retbins=retbins, 
                               precision=precision, duplicates=duplicates, **kwargs)
    pd.qcut = custom_qcut
    
    # DataFrame display extension
    pd.DataFrame.display = dataframe_display
    # Series display could be added if a similar styler exists or is desired
    # pd.Series.display = series_display 

    # Append extension
    pd.DataFrame.append = append # This might override built-in append, ensure behavior is as expected or rename
                                 # Pandas DataFrame.append is deprecated since 1.4.0 and will be removed in future. 
                                 # This custom append could be a replacement.

    # ggplot extension
    pd.DataFrame.ggplot = ggplot_df
    pd.Series.ggplot = ggplot_df

    _GLOBAL_INITIALIZED = True

# Optional: Call init_pandas_extensions() here if you want extensions to be applied when this module is imported.
# However, it's often better to explicitly call it from the main package __init__.py for clarity.
# init_pandas_extensions() 