# oequant

Straightforward algorithmic trading backtesting package.

[![Project Website](https://img.shields.io/badge/Project%20Website-Visit-blue?style=for-the-badge)](https://oequant.github.io/oequant/)

### Try it in Google Colab directly:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oequant/oequant/blob/main/examples/intro.ipynb)

## Features
*   Extremely simple to use!
*   Data fetching (currently `yfinance`)
*   Vectorized backtesting engine
*   Performance statistics calculation
*   Result plotting (Bokeh)
*   Signal generation via pandas expressions
*   Flexible position sizing
*   Benchmark comparison

## Installation

Currently, install directly from GitHub:

```bash
pip install git+https://github.com/oequant/oequant.git
```

(Requires Python >= 3.8)

## Usage

See the example notebook: [`examples/intro.ipynb`](./examples/intro.ipynb)

Here's a quick example of how to use `oequant`:

```python
import oequant as oq
import pandas_ta as ta

# 1. Get Data
ticker = "QQQ"
start_date = "2010-01-01"
end_date = "2025-05-01"
data = oq.get_data(ticker, start=start_date, end=end_date)

# 2. Add Indicators/Signals
# Calculate RSI_3
data.ta.rsi(length=3, append=True, col_names=('rsi_3',))
# Calculate exit signal: close > high.shift()
data['exit_signal'] = data['close'] > data['high'].shift(1)
# Entry signal: rsi_3 < 20 (will be passed as an expression)
entry_expression = "rsi_3 < 20"

# 3. Run Backtest
results = oq.backtest(
    data=data,
    entry_column=entry_expression,
    exit_column='exit_signal',
    capital=100_000,
    fee_frac=0.001
)

# 4. View Report
results.report(show_plot=True, plot_args={
    'indicators_other': ['rsi_3'],
    'per_indicator_plot_height': 160
})
```

This will produce a report similar to this:

```text
--- Backtest Report ---
Start Date:             2010-01-04
End Date:               2025-04-30
Duration:               5595 days 00:00:00
Initial Capital:        100,000.00
Final Equity:           211,555.28
Benchmark Final Equity: 1,174,411.72

--- Performance Metrics ---
|                    |   Strategy |   Benchmark |
|====================|============|=============|
|           CAGR (%) |      5.02% |      17.47% |
|      Avg Trade (%) |      0.47% |   1,074.41% |
|       Sharpe Ratio |     0.5031 |      0.8771 |
|      Sortino Ratio |     0.3253 |      1.1264 |
| Time in Market (%) |     15.93% |      99.97% |
|   Max Drawdown (%) |    -16.14% |     -35.12% |
|      CAGR / Max DD |     0.3111 |      0.4975 |
|     Serenity Ratio |     4.3518 |      3.1370 |
|       Total Trades |   169.0000 |      1.0000 |
|       Win Rate (%) |     69.23% |     100.00% |
|      Loss Rate (%) |     30.77% |       0.00% |
|  Avg Win Trade (%) |      1.52% |   1,074.41% |
| Avg Loss Trade (%) |     -1.90% |       0.00% |
|      Profit Factor |     1.7373 |         inf |
|      Avg Bars Held |     3.6331 |  3,854.0000 |
```

And an interactive plot like this:

![oequant plot example](docs/oequant_plot_example.png)

## License

This project is licensed under the terms of the MIT license with additional clauses. See the [LICENSE](./LICENSE) file.

## Contributing

Contributions are welcome! Please ensure you agree to the Contributor License Agreement outlined in the LICENSE file. 
