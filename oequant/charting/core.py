import pandas as pd
import numpy as np
from itertools import cycle

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    CrosshairTool,
    HoverTool,
    NumeralTickFormatter,
    DatetimeTickFormatter,
    LinearAxis,
    Range1d,
    Span,
    DataRange1d,
    GlyphRenderer
)
from bokeh.palettes import Category10
from bokeh.colors import RGB
from colorsys import hls_to_rgb, rgb_to_hls
from bokeh.themes import NIGHT_SKY, LIGHT_MINIMAL, CALIBER # Import themes
from bokeh.io import curdoc # Import curdoc

from oequant.backtesting.results import BacktestResult

BULL_COLOR = RGB(0, 255, 0)  # Lime
BEAR_COLOR = RGB(255, 0, 0) # Red
BENCH_COLOR = 'gray'        # Color for benchmark line

def _lightness(color, lightness=.94):
    rgb = np.array([color.r, color.g, color.b]) / 255
    h, _, s = rgb_to_hls(*rgb)
    rgb = (np.array(hls_to_rgb(h, lightness, s)) * 255).astype(int)
    return RGB(*rgb)

def _colorgen():
    yield from cycle(Category10[10])

def plot_results(
    result: BacktestResult,
    price_col: str = 'close',
    indicators_price: list = None,
    indicators_other: list = None,
    show_ohlc: bool = False,
    plot_width: int = 1000,
    main_price_plot_height: int = 400,       # Height for the main price chart
    per_indicator_plot_height: int = 80,     # Height for each 'other' indicator chart
    show_benchmark: bool = True,              # Whether to plot benchmark equity
    plot_theme: str = "dark"                  # Added plot_theme, default "dark"
) -> gridplot:
    """
    NOT USED CURRENTLY. plot_results_2 (bellow) is used instead.
    Keep for reference and notebooks compatibility for now.

    Plots the backtest results using Bokeh.

    Args:
        result (BacktestResult): The result object from a backtest.
        price_col (str): The column in ohlcv_data for main price plot (default 'close').
        indicators_price (list, optional): List of column names from ohlcv_data to plot on the main price chart.
        indicators_other (list, optional): List of column names from ohlcv_data to plot on separate subplots.
        show_ohlc (bool, optional): If True, attempts to plot OHLC data if available. Defaults to False.
        plot_width (int, optional): Width of the plot. Heights are calculated based on number of subplots.
        main_price_plot_height (int): Height of the main price plot.
        per_indicator_plot_height (int): Height of each secondary indicator plot.
        show_benchmark (bool): If True and benchmark results exist, plot benchmark equity. Defaults to True.
        plot_theme (str, optional): Theme for the plot ("dark", "light", or None for default Bokeh). Defaults to "dark".

    Returns:
        bokeh.layouts.LayoutDOM: The Bokeh gridplot layout object.
    """
    original_theme = curdoc().theme # Store original theme
    try:
        # ---- Theme Selection ----
        selected_theme_name = None # Store the name or Theme object
        if plot_theme == "dark":
            selected_theme_name = NIGHT_SKY
        elif plot_theme == "light":
            selected_theme_name = LIGHT_MINIMAL
        
        if selected_theme_name:
            curdoc().theme = selected_theme_name

        # Define background color based on theme for explicit setting
        dark_bg_color = "#2F2F2F"
        border_color = dark_bg_color if plot_theme == "dark" else None # Use dark for border too in dark mode
        bg_color = dark_bg_color if plot_theme == "dark" else None
        text_color = "white" if plot_theme == "dark" else "black" # Define text color

        ohlcv = result.ohlcv_data.copy()
        trades = result.trades.copy()
        returns = result.returns.copy()
        equity_curve = returns['equity']

        if indicators_price is None: indicators_price = []
        if indicators_other is None: indicators_other = []

        # Prepare data sources
        ohlcv.index.name = 'datetime' # Bokeh likes named index for DatetimeTickFormatter
        ohlcv = ohlcv.reset_index()
        source = ColumnDataSource(ohlcv)
        
        # Handle empty trades DataFrame
        if trades.empty:
            # Create an empty source with the expected columns but no data
            trade_data = {
                'entry_time': [], 'exit_time': [], 'entry_price': [], 'exit_price': [],
                'quantity': [], 'pnl_net': [], 'pnl_positive': []
            }
        else:
            trade_data = {
                'entry_time': trades['entry_time'],
                'exit_time': trades['exit_time'],
                'entry_price': trades['entry_price'],
                'exit_price': trades['exit_price'],
                'quantity': trades['quantity'],
                'pnl_net': trades['pnl_net_currency'],
                'pnl_positive': (trades['pnl_net_currency'] > 0).astype(int).astype(str)
            }
        trade_source = ColumnDataSource(trade_data)

        # Tools - Apply crosshair across linked plots
        tools = "xpan,xwheel_zoom,box_zoom,undo,redo,reset,save"
        crosshair = CrosshairTool(dimensions="both", line_color="gray", line_width=1)

        # Determine number of plots and heights
        num_other_indicators = len(indicators_other)

        # main_price_plot_height is for fig_price (from parameter)
        # per_indicator_plot_height is for each fig_indicator (from parameter)

        # Calculate equity curve height such that it's 1/3 of the sum of:
        # main_price_plot_height, calculated_equity_height, and (num_other_indicators * per_indicator_plot_height)
        # Derivation: H_e = (main_price_plot_height + num_other_indicators * per_indicator_plot_height) / 2
        calculated_equity_height = int(round((main_price_plot_height + num_other_indicators * per_indicator_plot_height) / 2))

        # Ensure a minimum reasonable height for the equity plot
        min_eq_height = max(50, int(per_indicator_plot_height * 0.75)) # e.g. 75% of an indicator height, or 50px
        if calculated_equity_height < min_eq_height:
            calculated_equity_height = min_eq_height
        
        # ---- Price Plot ----
        fig_price = figure(
            height=main_price_plot_height, width=plot_width, tools=tools,
            x_axis_type="datetime", x_axis_location="above",
            title=f"{price_col.capitalize()} with Trades",
            y_range=DataRange1d(only_visible=True),
            background_fill_color=bg_color, # Explicit background
            border_fill_color=border_color    # Explicit border
        )
        fig_price.yaxis.formatter = NumeralTickFormatter(format="0,0.0[00]")
        fig_price.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", days="%d %b") # Show month/year initially
        # Apply text colors for dark theme
        if plot_theme == "dark":
            fig_price.title.text_color = text_color
            if fig_price.xaxis:
                fig_price.xaxis.axis_label_text_color = text_color
                fig_price.xaxis.major_label_text_color = text_color
                fig_price.xaxis.major_tick_line_color = text_color
                fig_price.xaxis.minor_tick_line_color = text_color
                fig_price.xaxis.axis_line_color = text_color # Axis line itself
            if fig_price.yaxis:
                fig_price.yaxis.axis_label_text_color = text_color
                fig_price.yaxis.major_label_text_color = text_color
                fig_price.yaxis.major_tick_line_color = text_color
                fig_price.yaxis.minor_tick_line_color = text_color
                fig_price.yaxis.axis_line_color = text_color # Axis line itself
            if fig_price.legend:
                fig_price.legend.label_text_color = text_color
                # Potentially set legend title color if it exists
                # fig_price.legend.title_text_color = text_color 

        is_ohlc_available = show_ohlc and all(c in ohlcv.columns for c in ['open', 'high', 'low', 'close'])
        if is_ohlc_available:
            inc = ohlcv.close > ohlcv.open
            dec = ohlcv.open > ohlcv.close
            w = (ohlcv['datetime'].iloc[1] - ohlcv['datetime'].iloc[0]).total_seconds() * 1000 * 0.8 if len(ohlcv) > 1 else 86400000 * 0.8 # Bar width in ms

            fig_price.segment('datetime', 'high', 'datetime', 'low', source=source, color="black")
            fig_price.vbar('datetime', w, 'open', 'close', source=source, fill_color=BULL_COLOR, line_color="black", name="candlestick_inc", legend_label="OHLC")
            fig_price.vbar('datetime', w, 'open', 'close', source=source, fill_color=BEAR_COLOR, line_color="black", name="candlestick_dec", legend_label="OHLC")
            # Add filter views later if needed to hide based on inc/dec
        else:
            fig_price.line('datetime', price_col, source=source, legend_label=price_col.capitalize(), color='blue', alpha=0.8)

        # Plot price indicators
        price_indicator_colors = _colorgen()
        for indicator_name in indicators_price:
            if indicator_name in ohlcv.columns:
                fig_price.line('datetime', indicator_name, source=source, legend_label=indicator_name, color=next(price_indicator_colors))

        # Plot trades
        if not trades.empty:
            entry_markers = fig_price.scatter(
                'entry_time', 'entry_price', source=trade_source,
                marker='triangle', size=12, color='lime',
                legend_label=f"Trades ({len(trades)})")
            exit_markers = fig_price.scatter(
                'exit_time', 'exit_price', source=trade_source,
                marker='inverted_triangle', size=12, color='red',
                legend_label=f"Trades ({len(trades)})")
            # Tooltip for trades
            fig_price.add_tools(HoverTool(renderers=[entry_markers, exit_markers], tooltips=[
                ("Action", "Entry/Exit"), # Placeholder, maybe add type to trade_source
                ("Time", "@entry_time{%F %T}" if "entry_time" in trade_source.data else "@exit_time{%F %T}"), # Use formatter
                ("Price", "@entry_price{0,0.0[00]}" if "entry_price" in trade_source.data else "@exit_price{0,0.0[00]}"),
                ("Size", "@quantity{0,0.0[00]}"),
                ("PnL", "@pnl_net{0,0.0[00]}")
            ], formatters={'@entry_time': 'datetime', '@exit_time': 'datetime'}))

        fig_price.legend.location = "top_left"
        fig_price.legend.click_policy = "hide"
        fig_price.legend.background_fill_alpha = 0.8
        fig_price.add_tools(crosshair)

        # ---- Equity Curve Plot ----
        fig_equity = figure(
            height=calculated_equity_height, width=plot_width, tools=tools,
            x_range=fig_price.x_range, # Link x-axes
            x_axis_type="datetime",
            title="Equity Curve",
            y_range=DataRange1d(only_visible=True),
            background_fill_color=bg_color, # Explicit background
            border_fill_color=border_color    # Explicit border
        )
        returns_for_plot = returns.reset_index() # Index (e.g., 'date') becomes a column
        date_col_name = returns_for_plot.columns[0] # The first column is the former index
        
        # Apply text colors for dark theme
        if plot_theme == "dark":
            fig_equity.title.text_color = text_color
            # x-axis is initially hidden, but set y-axis
            if fig_equity.yaxis:
                fig_equity.yaxis.axis_label_text_color = text_color
                fig_equity.yaxis.major_label_text_color = text_color
                fig_equity.yaxis.major_tick_line_color = text_color
                fig_equity.yaxis.minor_tick_line_color = text_color
                fig_equity.yaxis.axis_line_color = text_color
            if fig_equity.legend:
                fig_equity.legend.label_text_color = text_color
                # fig_equity.legend.title_text_color = text_color

        # Add benchmark equity if available and requested
        if result.benchmark_res and show_benchmark:
            benchmark_equity = result.benchmark_res.returns['equity']
            # Align benchmark index with strategy returns index if necessary (though backtest should ensure same index)
            benchmark_equity = benchmark_equity.reindex(returns.index).reset_index()[equity_curve.name] # get Series
            returns_for_plot['benchmark_equity'] = benchmark_equity
            has_benchmark = True
        else:
            has_benchmark = False
            returns_for_plot['benchmark_equity'] = np.nan # Add column even if empty for consistent source
        
        equity_source = ColumnDataSource(returns_for_plot)
        
        # Plot Strategy Equity
        fig_equity.line(x=date_col_name, y='equity', source=equity_source, color='purple', legend_label="Strategy Equity")
        
        # Plot Benchmark Equity (if available)
        if has_benchmark:
            fig_equity.line(x=date_col_name, y='benchmark_equity', source=equity_source, color=BENCH_COLOR, line_dash='dashed', legend_label="Benchmark Equity", visible=False)
        
        fig_equity.yaxis.formatter = NumeralTickFormatter(format="$ 0,0")
        fig_equity.xaxis.visible = False
        fig_equity.legend.location = "top_left"
        fig_equity.legend.click_policy = "hide"
        fig_equity.add_tools(crosshair)
        
        # Define tooltips, conditionally add benchmark
        tooltips_equity = [("Date", f"@{date_col_name}{{%F}}"), ("Strategy Equity", "@equity{$0,0}")]
        if has_benchmark:
            tooltips_equity.append(("Benchmark Equity", "@benchmark_equity{$0,0}"))
        
        fig_equity.add_tools(HoverTool(tooltips=tooltips_equity, 
                                     formatters={f'@{date_col_name}': 'datetime'}, mode='vline'))

        # ---- Other Indicator Plots ----
        indicator_figs = []
        other_indicator_colors = _colorgen()
        for indicator_name in indicators_other:
            fig_indicator = figure(
                height=per_indicator_plot_height, width=plot_width, tools=tools,
                x_range=fig_price.x_range, # Link x-axes
                x_axis_type="datetime",
                title=indicator_name.replace('_', ' ').title(),
                y_range=DataRange1d(only_visible=True),
                background_fill_color=bg_color, # Explicit background
                border_fill_color=border_color    # Explicit border
            )
            r = fig_indicator.line(x='datetime', y=indicator_name, source=source, color=next(other_indicator_colors))
            fig_indicator.yaxis.formatter = NumeralTickFormatter(format="0,0.0[00]")
            fig_indicator.xaxis.visible = False # Initially hidden
            # Apply text colors for dark theme
            if plot_theme == "dark":
                fig_indicator.title.text_color = text_color
                # x-axis is initially hidden, but set y-axis
                if fig_indicator.yaxis:
                    fig_indicator.yaxis.axis_label_text_color = text_color
                    fig_indicator.yaxis.major_label_text_color = text_color
                    fig_indicator.yaxis.major_tick_line_color = text_color
                    fig_indicator.yaxis.minor_tick_line_color = text_color
                    fig_indicator.yaxis.axis_line_color = text_color
                # No legend usually on these individual indicators
            fig_indicator.add_tools(crosshair)
            fig_indicator.add_tools(HoverTool(renderers=[r], tooltips=[
                ("Date", "@datetime{%F}"), 
                (indicator_name, f"@{indicator_name}{{0,0.0[00]}}")
            ], formatters={'@datetime': 'datetime'}, mode='vline'))
            indicator_figs.append(fig_indicator)

        # ---- Combine Plots ----
        # Show final x-axis on the last plot
        last_plot = indicator_figs[-1] if indicator_figs else fig_equity
        last_plot.xaxis.visible = True
        last_plot.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", days="%d %b")
        # Apply text colors to the now-visible last x-axis if dark theme
        if plot_theme == "dark":
            if last_plot.xaxis:
                last_plot.xaxis.axis_label_text_color = text_color
                last_plot.xaxis.major_label_text_color = text_color
                last_plot.xaxis.major_tick_line_color = text_color
                last_plot.xaxis.minor_tick_line_color = text_color
                last_plot.xaxis.axis_line_color = text_color

        layout = gridplot([[fig_price], [fig_equity]] + [[fig] for fig in indicator_figs], 
                          toolbar_location="right", merge_tools=True)

        return layout
    finally:
        curdoc().theme = original_theme # Restore original theme 

def plot_results_2(
    result: BacktestResult,
    price_col: str = 'close',
    indicators_price: list = None,
    indicators_other: list = None,
    show_ohlc: bool = True,
    plot_width: int = 1000,
    main_price_plot_height: int = 300,       # Adjusted default height
    equity_plot_height: int = 150,           # Height for equity
    pnl_plot_height: int = 120,              # Height for P&L plot
    volume_plot_height: int = 75,           # Height for volume plot (default changed from 100)
    per_indicator_plot_height: int = 80,     # Height for each 'other' indicator chart
    show_benchmark: bool = True,
    plot_theme: str = "dark"
) -> gridplot:
    """
    Plots the backtest results using Bokeh, including a P&L and Volume subplot.
    Inspired by plot_results and common financial charting layouts.

    Args:
        result (BacktestResult): The result object from a backtest.
        price_col (str): Column in ohlcv_data for main price plot (default 'close').
        indicators_price (list, optional): List of column names from ohlcv_data to plot on the main price chart.
        indicators_other (list, optional): List of column names from ohlcv_data to plot on separate subplots.
        show_ohlc (bool, optional): If True, attempts to plot OHLC data. Defaults to True.
        plot_width (int, optional): Width of the plot.
        main_price_plot_height (int): Height of the main price plot.
        equity_plot_height (int): Height of the equity curve plot.
        pnl_plot_height (int): Height of the Profit/Loss subplot.
        volume_plot_height (int): Height of the volume subplot.
        per_indicator_plot_height (int): Height of each secondary indicator plot.
        show_benchmark (bool): If True and benchmark results exist, plot benchmark equity. Defaults to True.
        plot_theme (str, optional): Theme for the plot ("dark", "light", or None). Defaults to "dark".

    Returns:
        bokeh.layouts.gridplot: The Bokeh gridplot layout object.
    """
    original_theme = curdoc().theme
    try:
        selected_theme_name = None
        if plot_theme == "dark":
            selected_theme_name = NIGHT_SKY
        elif plot_theme == "light":
            selected_theme_name = LIGHT_MINIMAL
        
        if selected_theme_name:
            curdoc().theme = selected_theme_name

        dark_bg_color = "#2F2F2F"
        border_color = dark_bg_color if plot_theme == "dark" else None
        bg_color = dark_bg_color if plot_theme == "dark" else None
        text_color = "white" if plot_theme == "dark" else "black"
        grid_line_color = "gray" if plot_theme == "dark" else "#e0e0e0"
        axis_line_color = text_color


        ohlcv = result.ohlcv_data.copy()
        trades = result.trades.copy()
        returns = result.returns.copy()
        equity_curve = returns['equity']

        if indicators_price is None: indicators_price = []
        if indicators_other is None: indicators_other = []

        ohlcv.index.name = 'datetime'
        ohlcv_source_df = ohlcv.reset_index()
        
        # Prepare volume colors and add to ohlcv_source_df before creating ColumnDataSource
        if 'open' in ohlcv_source_df.columns and 'close' in ohlcv_source_df.columns:
            ohlcv_source_df['volume_bar_colors'] = [BULL_COLOR if close_price > open_price else BEAR_COLOR for open_price, close_price in zip(ohlcv_source_df['open'], ohlcv_source_df['close'])]
        else:
            # Fallback color if open/close are not available (e.g., single price series data)
            ohlcv_source_df['volume_bar_colors'] = 'silver' # Assign a single color, Bokeh handles this
            
        source = ColumnDataSource(ohlcv_source_df)
        
        trade_fields = {
            'entry_time': [], 'exit_time': [], 'entry_price': [], 'exit_price': [],
            'quantity': [], 'pnl_net_curr': [], 'pnl_net_frac': [], 'pnl_positive_color': []
        }
        if not trades.empty:
            trade_fields['entry_time'] = trades['entry_time']
            trade_fields['exit_time'] = trades['exit_time']
            trade_fields['entry_price'] = trades['entry_price']
            trade_fields['exit_price'] = trades['exit_price']
            trade_fields['quantity'] = trades['quantity']
            trade_fields['pnl_net_curr'] = trades['pnl_net_currency']
            trade_fields['pnl_net_frac'] = trades['pnl_net_frac'] * 100 # Convert to percentage for P&L plot
            trade_fields['pnl_positive_color'] = trades['pnl_net_currency'].apply(lambda x: BULL_COLOR if x > 0 else (BEAR_COLOR if x < 0 else 'gray'))
        trade_source = ColumnDataSource(trade_fields)

        tools = "xpan,xwheel_zoom,box_zoom,undo,redo,reset,save"
        crosshair = CrosshairTool(dimensions="both", line_color="gray", line_width=1)
        
        plot_figures = []
        left_border_padding = 40 # Increased padding for vertical labels

        # ---- Equity Curve Plot ----
        stats = result.statistics() # Get all stats
        equity_source_df = returns.reset_index().copy()
        date_col_name = equity_source_df.columns[0]
        equity_source_df['equity_cummax'] = equity_source_df['equity'].cummax() # For drawdown patch
        equity_cds = ColumnDataSource(equity_source_df)

        fig_equity = figure(
            height=equity_plot_height, width=plot_width, tools=tools,
            x_axis_type="datetime", # title="Equity Curve", Removed title
            y_range=DataRange1d(only_visible=True, range_padding=0.01),
            background_fill_color=bg_color, border_fill_color=border_color
        )
        fig_equity.min_border_left = left_border_padding
        fig_equity.yaxis.axis_label = "Equity Curve" # Set as Y-axis label
        fig_equity.yaxis.axis_label_orientation = "vertical" # Rotate label
        
        # Drawdown patch (area between equity and high-water mark)
        patch_x_data = np.concatenate((equity_source_df[date_col_name].values, equity_source_df[date_col_name].values[::-1]))
        patch_y_data = np.concatenate((equity_source_df['equity'].values, equity_source_df['equity_cummax'].values[::-1]))
        fig_equity.patch(patch_x_data, patch_y_data, 
                         fill_color=RGB(255, 230, 230, 0.4) if plot_theme == "dark" else RGB(255, 230, 230, 0.6), 
                         line_color=RGB(230, 150, 150, 0.4) if plot_theme == "dark" else RGB(230,150,150,0.6), 
                         alpha=0.7)

        equity_line = fig_equity.line(date_col_name, 'equity', source=equity_cds, color="skyblue", legend_label="Strategy Equity", line_width=1.5)

        if result.benchmark_res and show_benchmark:
            benchmark_equity_series = result.benchmark_res.returns['equity'].reindex(returns.index).reset_index()['equity']
            equity_source_df['benchmark_equity'] = benchmark_equity_series # Add to the df for CDS update
            equity_cds.data = ColumnDataSource.from_df(equity_source_df) # Update CDS
            fig_equity.line(date_col_name, 'benchmark_equity', source=equity_cds, color=BENCH_COLOR, legend_label="Benchmark Equity", line_dash="dashed", visible=False)

        # Annotations
        peak_equity_val = equity_curve.max()
        peak_equity_idx = equity_curve.idxmax()
        if pd.notna(peak_equity_val) and pd.notna(peak_equity_idx):
            fig_equity.scatter(peak_equity_idx, peak_equity_val, legend_label=f'Peak ({peak_equity_val:,.0f})', color='cyan', size=8, name="peak_equity_marker")

        final_equity_val = equity_curve.iloc[-1]
        final_equity_idx = equity_curve.index[-1]
        if pd.notna(final_equity_val) and pd.notna(final_equity_idx):
            fig_equity.scatter(final_equity_idx, final_equity_val, legend_label=f'Final ({final_equity_val:,.0f})', color='blue', size=8, name="final_equity_marker")

        max_dd_pct_val = stats.get('max_dd_pct')
        max_dd_valley_date = stats.get('max_drawdown_valley_date')
        if pd.notna(max_dd_pct_val) and pd.notna(max_dd_valley_date) and max_dd_valley_date in equity_curve.index:
            max_dd_equity_at_valley = equity_curve.loc[max_dd_valley_date]
            fig_equity.scatter(max_dd_valley_date, max_dd_equity_at_valley, legend_label=f'Max Drawdown ({max_dd_pct_val:.1f}%)', color='red', size=8, name="max_dd_value_marker")

        max_dd_duration_days = stats.get('max_drawdown_duration_days')
        dd_line_start_date = stats.get('max_drawdown_start_date_of_longest_dd')
        dd_line_end_date = stats.get('max_drawdown_end_date_of_longest_dd')
        if pd.notna(max_dd_duration_days) and pd.notna(dd_line_start_date) and pd.notna(dd_line_end_date) and \
           dd_line_start_date in equity_curve.index and dd_line_end_date in equity_curve.index:
            equity_at_dd_line_start = equity_curve.loc[dd_line_start_date]
            fig_equity.line([dd_line_start_date, dd_line_end_date], 
                            [equity_at_dd_line_start, equity_at_dd_line_start], 
                            line_color='tomato', line_width=2, line_dash='solid', # Solid red line for duration
                            legend_label=f'Max DD Dur. ({max_dd_duration_days:.0f} days)', name="max_dd_duration_line")

        fig_equity.yaxis.formatter = NumeralTickFormatter(format="0,0")
        fig_equity.xaxis.visible = False # Hide x-axis labels for linked plots above price
        fig_equity.legend.location = "top_left"
        fig_equity.legend.click_policy = "hide"
        fig_equity.add_tools(crosshair)
        if plot_theme == "dark":
            fig_equity.title.text_color = text_color
            fig_equity.yaxis.axis_label_text_color = text_color
            fig_equity.yaxis.major_label_text_color = text_color
            fig_equity.yaxis.major_tick_line_color = text_color
            fig_equity.yaxis.minor_tick_line_color = text_color
            fig_equity.yaxis.axis_line_color = axis_line_color
            fig_equity.xgrid.grid_line_color = grid_line_color
            fig_equity.ygrid.grid_line_color = grid_line_color
            if fig_equity.legend: 
                fig_equity.legend.label_text_color = text_color
                fig_equity.legend.background_fill_color = dark_bg_color # Set dark background
                fig_equity.legend.background_fill_alpha = 0.5 # Set alpha to 0.5
        plot_figures.append(fig_equity)

        # ---- Profit/Loss Plot ----
        fig_pnl = figure(
            height=pnl_plot_height, width=plot_width, tools=tools,
            x_range=fig_equity.x_range, x_axis_type="datetime",
            # title="Profit / Loss per Trade (%)", Removed title
            y_range=DataRange1d(only_visible=True, range_padding=0.01),
            background_fill_color=bg_color, border_fill_color=border_color
        )
        fig_pnl.min_border_left = left_border_padding
        fig_pnl.yaxis.axis_label = "P/L (%)" # Set as Y-axis label (shortened)
        fig_pnl.yaxis.axis_label_orientation = "vertical" # Rotate label
        fig_pnl.segment(x0='exit_time', y0=0, x1='exit_time', y1='pnl_net_frac', source=trade_source, line_width=2, color='pnl_positive_color', alpha=0.7)
        fig_pnl.scatter(x='exit_time', y='pnl_net_frac', source=trade_source, size=8, color='pnl_positive_color', legend_label="Trade P&L")
        
        zero_line = Span(location=0, dimension='width', line_color='gray', line_dash='dashed', line_width=1)
        fig_pnl.add_layout(zero_line)
        fig_pnl.yaxis.formatter = NumeralTickFormatter(format="0.0'%'")
        fig_pnl.xaxis.visible = False
        fig_pnl.legend.location = "top_left"
        fig_pnl.legend.click_policy = "hide"
        fig_pnl.add_tools(crosshair)
        if plot_theme == "dark":
            fig_pnl.title.text_color = text_color
            fig_pnl.yaxis.axis_label_text_color = text_color
            fig_pnl.yaxis.major_label_text_color = text_color
            fig_pnl.yaxis.major_tick_line_color = text_color
            fig_pnl.yaxis.minor_tick_line_color = text_color
            fig_pnl.yaxis.axis_line_color = axis_line_color
            fig_pnl.xgrid.grid_line_color = grid_line_color
            fig_pnl.ygrid.grid_line_color = grid_line_color
            if fig_pnl.legend: 
                fig_pnl.legend.label_text_color = text_color
                fig_pnl.legend.background_fill_color = dark_bg_color # Set dark background
                fig_pnl.legend.background_fill_alpha = 0.5 # Set alpha to 0.5
        
        pnl_hover = HoverTool(renderers=fig_pnl.select(type=GlyphRenderer), tooltips=[
            ("Exit Time", "@exit_time{%F %T}"),
            ("P&L (%)", "@pnl_net_frac{0.00}%"),
            ("P&L ($)", "@pnl_net_curr{0,0.00}")
        ], formatters={'@exit_time': 'datetime'})
        fig_pnl.add_tools(pnl_hover)
        plot_figures.append(fig_pnl)

        # ---- Price Plot ----
        fig_price = figure(
            height=main_price_plot_height, width=plot_width, tools=tools,
            x_range=fig_equity.x_range, x_axis_type="datetime", # x_axis_location="above", Removed this
            # title=f"{price_col.capitalize()} Chart with Trades", Removed title
            y_range=DataRange1d(only_visible=True, range_padding=0.01),
            background_fill_color=bg_color, border_fill_color=border_color
        )
        fig_price.min_border_left = left_border_padding
        fig_price.yaxis.axis_label = price_col.capitalize() # Set as Y-axis label
        fig_price.yaxis.axis_label_orientation = "vertical" # Rotate label
        fig_price.xaxis.visible = False # Explicitly hide the x-axis
        
        is_ohlc_available = show_ohlc and all(c in ohlcv_source_df.columns for c in ['open', 'high', 'low', 'close'])
        if is_ohlc_available:
            w = (ohlcv_source_df['datetime'].iloc[1] - ohlcv_source_df['datetime'].iloc[0]).total_seconds() * 1000 * 0.8 if len(ohlcv_source_df) > 1 else 86400000 * 0.8
            fig_price.segment('datetime', 'high', 'datetime', 'low', source=source, color="black", name="ohlc_wick")
            inc = ohlcv_source_df.close > ohlcv_source_df.open
            dec = ohlcv_source_df.open > ohlcv_source_df.close
            source_inc = ColumnDataSource(ohlcv_source_df[inc])
            source_dec = ColumnDataSource(ohlcv_source_df[dec])
            fig_price.vbar('datetime', w, 'open', 'close', source=source_inc, fill_color=BULL_COLOR, line_color="black", legend_label="OHLC Up", name="ohlc_up")
            fig_price.vbar('datetime', w, 'open', 'close', source=source_dec, fill_color=BEAR_COLOR, line_color="black", legend_label="OHLC Down", name="ohlc_down")
        else:
            fig_price.line('datetime', price_col, source=source, legend_label=price_col.capitalize(), color='skyblue', alpha=0.8)

        price_indicator_colors = _colorgen()
        for indicator_name in indicators_price:
            if indicator_name in ohlcv_source_df.columns:
                fig_price.line('datetime', indicator_name, source=source, legend_label=indicator_name, color=next(price_indicator_colors))

        if not trades.empty:
            entry_markers = fig_price.scatter(
                'entry_time', 'entry_price', source=trade_source,
                marker='triangle', size=10, color='lime', alpha=0.7, legend_label=f"Entries ({len(trades)})")
            exit_markers = fig_price.scatter(
                'exit_time', 'exit_price', source=trade_source,
                marker='inverted_triangle', size=10, color='red', alpha=0.7, legend_label=f"Exits ({len(trades)})")
            trade_hover = HoverTool(renderers=[entry_markers, exit_markers], tooltips=[
                ("Time", "@exit_time{%F %T}"), ("Price", "@exit_price{0,0.00}"),
                ("Size", "@quantity{0,0.00}"), ("PnL", "@pnl_net_curr{0,0.00}")
            ], formatters={'@exit_time': 'datetime'}) # Assume exit_time always present
            fig_price.add_tools(trade_hover)
        
        fig_price.legend.location = "top_left"
        fig_price.legend.click_policy = "hide"
        fig_price.add_tools(crosshair)
        
        # Apply dark theme styling to legend AFTER glyphs are added
        if plot_theme == "dark":
            fig_price.title.text_color = text_color
            # X-axis styling (though it's hidden, style for consistency if made visible)
            fig_price.xaxis.axis_label_text_color = text_color
            fig_price.xaxis.major_label_text_color = text_color
            fig_price.xaxis.major_tick_line_color = text_color
            fig_price.xaxis.minor_tick_line_color = text_color
            fig_price.xaxis.axis_line_color = axis_line_color
            # Y-axis styling
            fig_price.yaxis.axis_label_text_color = text_color
            fig_price.yaxis.major_label_text_color = text_color
            fig_price.yaxis.major_tick_line_color = text_color
            fig_price.yaxis.minor_tick_line_color = text_color
            fig_price.yaxis.axis_line_color = axis_line_color
            fig_price.xgrid.grid_line_color = grid_line_color
            fig_price.ygrid.grid_line_color = grid_line_color
            if fig_price.legend: 
                fig_price.legend.label_text_color = text_color
                fig_price.legend.background_fill_color = dark_bg_color # Set dark background
                fig_price.legend.background_fill_alpha = 0.5 # Set alpha to 0.5
        
        plot_figures.append(fig_price)

        # ---- Volume Plot ----
        if 'volume' in ohlcv_source_df.columns:
            fig_volume = figure(
                height=volume_plot_height, width=plot_width, tools=tools,
                x_range=fig_equity.x_range, x_axis_type="datetime",
                # title="Volume", Removed title
                y_range=DataRange1d(only_visible=True, range_padding=0.01),
                background_fill_color=bg_color, border_fill_color=border_color
            )
            fig_volume.min_border_left = left_border_padding
            fig_volume.yaxis.axis_label = "Volume" # Set as Y-axis label
            fig_volume.yaxis.axis_label_orientation = "vertical" # Rotate label
            fig_volume.vbar('datetime', (ohlcv_source_df['datetime'].iloc[1] - ohlcv_source_df['datetime'].iloc[0]).total_seconds() * 1000 * 0.8 if len(ohlcv_source_df) > 1 else 86400000 * 0.8, 
                            'volume', source=source, color='volume_bar_colors', alpha=0.5) # Use column name for color
            fig_volume.yaxis.formatter = NumeralTickFormatter(format="0.0a")
            fig_volume.xaxis.visible = False # Hide for all but last plot
            fig_volume.add_tools(crosshair)
            if plot_theme == "dark":
                fig_volume.title.text_color = text_color
                fig_volume.yaxis.axis_label_text_color = text_color
                fig_volume.yaxis.major_label_text_color = text_color
                fig_volume.yaxis.major_tick_line_color = text_color
                fig_volume.yaxis.minor_tick_line_color = text_color
                fig_volume.yaxis.axis_line_color = axis_line_color
                fig_volume.xgrid.grid_line_color = grid_line_color
                fig_volume.ygrid.grid_line_color = grid_line_color
            plot_figures.append(fig_volume)

        # ---- Other Indicator Plots ----
        other_indicator_colors = _colorgen()
        for i, indicator_name in enumerate(indicators_other):
            if indicator_name in ohlcv_source_df.columns:
                fig_indicator = figure(
                    height=per_indicator_plot_height, width=plot_width, tools=tools,
                    x_range=fig_equity.x_range, x_axis_type="datetime",
                    # title=indicator_name, Removed title
                    y_range=DataRange1d(only_visible=True, range_padding=0.01),
                    background_fill_color=bg_color, border_fill_color=border_color
                )
                fig_indicator.min_border_left = left_border_padding
                fig_indicator.yaxis.axis_label = indicator_name # Set as Y-axis label
                fig_indicator.yaxis.axis_label_orientation = "vertical" # Rotate label
                r = fig_indicator.line('datetime', indicator_name, source=source, color=next(other_indicator_colors))
                fig_indicator.yaxis.formatter = NumeralTickFormatter(format="0,0.0[00]")
                fig_indicator.xaxis.visible = (i == len(indicators_other) - 1) # Only last indicator plot shows x-axis if no volume plot
                
                if plot_theme == "dark":
                    fig_indicator.title.text_color = text_color
                    # X-axis styling
                    fig_indicator.xaxis.axis_label_text_color = text_color
                    fig_indicator.xaxis.major_label_text_color = text_color
                    fig_indicator.xaxis.major_tick_line_color = text_color
                    fig_indicator.xaxis.minor_tick_line_color = text_color
                    fig_indicator.xaxis.axis_line_color = axis_line_color
                    # Y-axis styling
                    fig_indicator.yaxis.axis_label_text_color = text_color
                    fig_indicator.yaxis.major_label_text_color = text_color
                    fig_indicator.yaxis.major_tick_line_color = text_color
                    fig_indicator.yaxis.minor_tick_line_color = text_color
                    fig_indicator.yaxis.axis_line_color = axis_line_color
                    fig_indicator.xgrid.grid_line_color = grid_line_color
                    fig_indicator.ygrid.grid_line_color = grid_line_color

                hover_indicator = HoverTool(renderers=[r], tooltips=[
                    (indicator_name, f"@{indicator_name}{{0,0.0[000]}}"),
                    ("Time", "$x{%F %T}")
                ], formatters={'$x': 'datetime'}, mode='vline')
                fig_indicator.add_tools(hover_indicator, crosshair)
                plot_figures.append(fig_indicator)

        # Make the x-axis visible for the last plot in the list
        if plot_figures:
            last_plot = plot_figures[-1]
            last_plot.xaxis.visible = True
            last_plot.xaxis.formatter = DatetimeTickFormatter(months="%b '%y", days="%d %b")
            if plot_theme == "dark": # Ensure last x-axis text is visible in dark mode
                last_plot.xaxis.axis_label_text_color = text_color
                last_plot.xaxis.major_label_text_color = text_color
                last_plot.xaxis.major_tick_line_color = text_color
                last_plot.xaxis.minor_tick_line_color = text_color
                last_plot.xaxis.axis_line_color = axis_line_color


        grid_layout = gridplot(plot_figures, ncols=1, sizing_mode='stretch_width') # Use gridplot for better layout control
        
        return grid_layout

    finally:
        curdoc().theme = original_theme # Restore original theme 