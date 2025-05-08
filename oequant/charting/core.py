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
    DataRange1d
)
from bokeh.palettes import Category10
from bokeh.colors import RGB
from colorsys import hls_to_rgb, rgb_to_hls

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
    show_benchmark: bool = True              # Whether to plot benchmark equity
) -> gridplot:
    """
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

    Returns:
        bokeh.layouts.LayoutDOM: The Bokeh gridplot layout object.
    """
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
        y_range=DataRange1d(only_visible=True)
    )
    fig_price.yaxis.formatter = NumeralTickFormatter(format="0,0.0[00]")
    fig_price.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", days="%d %b") # Show month/year initially

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
        y_range=DataRange1d(only_visible=True)
    )
    # Prepare returns data for Bokeh source
    returns_for_plot = returns.reset_index() # Index (e.g., 'date') becomes a column
    date_col_name = returns_for_plot.columns[0] # The first column is the former index
    
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
            y_range=DataRange1d(only_visible=True)
        )
        r = fig_indicator.line(x='datetime', y=indicator_name, source=source, color=next(other_indicator_colors))
        fig_indicator.yaxis.formatter = NumeralTickFormatter(format="0,0.0[00]")
        fig_indicator.xaxis.visible = False
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
    
    layout = gridplot([[fig_price], [fig_equity]] + [[fig] for fig in indicator_figs], 
                      toolbar_location="right", merge_tools=True)

    return layout 