# Stock Analyses Application

A Python application for analyzing and visualizing financial instruments data. This tool helps investors and financial analysts examine stock performance metrics, calculate risk and return ratios, and make data-driven investment decisions.

![Risk-Return Chart](https://github.com/username/Stock-Analyses-Application/raw/main/examples/risk_return_chart.png)

## Features

- **Individual Stock Analysis**: Detailed examination of a single financial instrument
- **Financial Metrics Calculation**:
  - Log returns calculation
  - Investment multiple calculation
  - Price charts and visualizations
  - Risk and return analysis
- **Multi-Stock Comparison**: Compare performance metrics across multiple stocks
- **Historical Data Analysis**: Analyze stock performance over custom time periods
- **Data Visualization**: Generate informative charts to visualize financial data

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/username/Stock-Analyses-Application.git
cd Stock-Analyses-Application
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Stock Analysis

```python
from app.FinancialInstrunment import FinancialInstrunment as FI

# Initialize a financial instrument with ticker symbol and date range
instrument = FI('AAPL', '2020-01-01', '2023-12-31')

# Plot price chart
instrument.plot_prices()

# Calculate investment multiple
multiple = instrument.investment_multiple()
print(f"Investment multiple: {multiple}")
```

### Comparing Multiple Stocks

```python
from app.InstrunmentComparitor import InstrunmentComparitor as IC

# Define tickers and date range
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
start_date = '2020-01-01'
end_date = '2023-12-31'

# Initialize the comparator
comparator = IC(tickers, start_date, end_date)

# Generate risk/reward comparison
comparator.compare_risk_reward()
```

### Running the Interactive Application

```bash
python app/main.py
```

Follow the prompts to enter:
- Ticker symbol (from Yahoo Finance)
- Start date (YYYY-MM-DD format)
- End date (YYYY-MM-DD format)

## Data Sources

The application uses [Yahoo Finance](https://finance.yahoo.com/) as its primary data source for stock information.

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Data visualization
- yfinance: Yahoo Finance API wrapper
- scipy: Statistical functions

## License

This project is licensed under the Mozilla Public License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## TODO

- Add annualized performance method
- Add skewness calculation method
- Add kurtosis calculation method
- Add rolling statistics (EMA 7/21/50 ....)
- Add CAGR (Compound Annual Growth Rate) method