import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # optional may be helpful for plotting percentage
import numpy as np
import pandas as pd
import seaborn as sb # optional to set plot theme
import pandas_datareader.data as web
sb.set_theme() # optional to set plot theme

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(days=10*365))
DEFAULT_END = dt.date.isoformat(dt.date.today())

class GDPData:
    def __init__(self, symbol="GDP", start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Fetch historical GDP data from FRED using pandas-datareader."""
        print(f"Fetching data for symbol: {self.symbol} from {self.start} to {self.end}")
        data = web.DataReader(self.symbol, "fred", self.start, self.end)
        data.index = pd.to_datetime(data.index)
        self.calc_returns(data)
        return data

    def calc_returns(self, df):
        """Calculate quarterly GDP changes and rate of return."""
        df['change'] = df[self.symbol].diff()  # Difference between quarters
        df['instant_return'] = np.log(df[self.symbol]).diff().round(4)  # Rate of return from one quarter to another
        self.data = df  # Update data with new columns

    def plot_return_dist(self):
        """Plot a histogram of GDP instantaneous returns."""
        plt.figure(figsize=(10, 5))
        sb.histplot(self.data['instant_return'].dropna(), bins=20, kde=True)
        plt.xlabel('Instantaneous Return')
        plt.ylabel('Frequency')
        plt.title(f'{self.symbol} Return Distribution')
        plt.show()

    def plot_performance(self):
        """Plot GDP performance over time as percentage change."""
        plt.figure(figsize=(12, 6))
        performance = (self.data[self.symbol] / self.data[self.symbol].iloc[0]) - 1
        plt.plot(self.data.index, performance * 100, label=f'{self.symbol} Performance')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Percent Change')
        plt.title(f'{self.symbol} Performance Over Time')
        plt.legend()
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.show()


def main():
    test_symbol = "GDP"  # Define test symbol
    test_gdp = GDPData(symbol=test_symbol)  # Instantiate class with GDP symbol
    print(f"Symbol: {test_gdp.symbol}")  # Print symbol
    print(f"Start Date: {test_gdp.start}")  # Print start date
    print(f"End Date: {test_gdp.end}")  # Print end date
    print(test_gdp.data.head())  # Display first few rows of GDP data
    test_gdp.plot_performance()  # Generate GDP performance plot
    test_gdp.plot_return_dist()  # Generate GDP return distribution plot

if __name__ == '__main__':
    main()
