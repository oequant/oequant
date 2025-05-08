# oequant

A simple quantitative trading backtesting package.


### Try it in Google Colab directly:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oequant/oequant/blob/main/examples/intro.ipynb)

## Features

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

```python
import oequant as oq
import pandas_ta as ta

# 1. Get Data
data = oq.get_data("SPY", start="2020-01-01")

# 2. Add Indicators/Signals
data.ta.rsi(length=14, append=True) # Adds RSI_14 column
data['entry'] = data['RSI_14'] < 30
data['exit'] = data['RSI_14'] > 70

# 3. Run Backtest (using 50% equity per trade)
results = oq.backtest(
    data,
    entry_column='entry',
    exit_column='exit',
    size=0.5, 
    size_unit='fraction',
    fee_frac=0.001 # 0.1% fee
)

# 4. View Report
results.report()
```

## License

This project is licensed under the terms of the MIT license with additional clauses. See the [LICENSE](./LICENSE) file.

## Contributing

Contributions are welcome! Please ensure you agree to the Contributor License Agreement outlined in the LICENSE file. 
