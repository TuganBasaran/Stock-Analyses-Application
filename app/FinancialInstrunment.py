import pandas as pd 
import numpy as np 
import yfinance as yf
import scipy.stats as stats
import matplotlib.pyplot as plt 

'''
    Application for financial instrunment analysis 
'''

'''
TODO:
- Add annualized performance method 
- Add skewness calculation method 
- Add kurtosisi calculation method 
- Add rolling statistics (EMA 7/21/50 ....) 
- Add Investment Multiple and CAGR (Compound Annual Growth Rate) method 
'''

class FinancialInstrunment(): 

    def __init__(self, ticker, start, end):
        '''Constructor for Financial Instrunment'''
        self.ticker = ticker 
        self.start = start 
        self.end = end 
        self.price_name = f"{self.ticker}_price"
        self.log_return_name = f"{self.ticker}_log_return"
        self.get_data()
        self.log_returns()
        

    def __repr__(self):
        return "Financial Instrunment (ticker: {}, start: {}, end: {})".format(self.ticker, self.start, self.end)
    
    def get_data(self): 
        ''' Downloading the data from the yahoo finance, renaming the ticker "close" name to "price".
        '''
        price_name = "{}_price".format(self.ticker)
        raw = yf.download(tickers= self.ticker, start= self.start, end= self.end, multi_level_index=False).Close.to_frame()
        raw.rename(columns={"Close" : price_name}, inplace=True)
        self.data = raw 

    def log_returns(self): 
        ''' Creating logarithmic daily returns'''
        self.data[self.log_return_name] = np.log(self.data / self.data.shift(1))

    def _setup_plot(self, title): 
        ''' Creating a price chart using line'''
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.plot(data=self.data, figsize=(15,8))
        plt.title(f"{self.ticker} {title}")
        plt.show()

    def investment_multiple(self): 
        '''Calculating the investment multiple from start date to end date'''
        start_price = self.data.iloc[0] 
        end_price = self.data.iloc[-1]
        multiple = round(end_price / start_price, 3)
        print(f"Investment multiple: {multiple}")
        return multiple


    def plot_prices(self): 
        print(f"Plotting {self.ticker}")
        price_name = "{}_price".format(self.ticker)
        self.data[price_name].plot(figsize= (12,8))
        self._setup_plot("Price Chart")
    
    def mean_return(self, freq=None): 
        if freq is None: 
            return self.data[self.log_return_name].mean()
        else: 
            resampled_price = self.data[self.price_name].resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
        return resampled_returns.mean()

    def std_risk(self, freq=None): 
        if freq is None: 
            return self.data[self.log_return_name].std()
        else: 
            resampled_price = self.data[self.price_name].resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()

    

    def plot_returns(self, freq=None, kind="hist"): 
        plt.figure(figsize=(12, 8))
        if freq is None: 
            returns = self.data[self.log_return_name]
        else: 
            resampled_price = self.data[self.price_name].resample(freq).last()
            returns = np.log(resampled_price / resampled_price.shift(1))

        # Normal dağılım ve histogram aynı figure üzerinde
        mu = returns.mean()
        sigma = returns.std()
        x = np.linspace(returns.min(), returns.max(), 10000)
        y = stats.norm.pdf(x, loc=mu, scale=sigma)
        plt.hist(returns, bins=500, density=True, label=f"Returns ({self.ticker})")
        plt.plot(x, y, linewidth=3, color="red", label="Normal Distribution")
        plt.title(f"{self.ticker} Returns Distribution", fontsize=20)
        plt.xlabel("Returns", fontsize=15)
        plt.ylabel("Density", fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.show()
    
    def std_mean_frame_ann(self): 
        '''
        Creates an standard deviation and mean scatter graph by time period
        Allowing trader to decide which time period is efficient 
        '''
        freqs = ["YE", "QE", "ME", "W-FRI", "D"]
        periods = [1, 4, 12, 52, 252]
        mean_list = []
        std_list = []
        for i in range(5): 
            resampled_data = self.data[self.price_name].resample(freqs[i]).last()
            resampled_return = np.log(resampled_data / resampled_data.shift(1))
            mean = resampled_return.mean() * periods[i]
            mean_list.append(mean)
            std =  resampled_return.std() * np.sqrt(periods[i])
            std_list.append(std)
        frame = pd.DataFrame(data={"Std_Risk": std_list,"Mean_Reward": mean_list}, index=freqs)    
        frame.plot(kind="scatter", x = "Std_Risk", y="Mean_Reward", figsize=(15,8), s=50, fontsize=15)
        for i in frame.index: 
            plt.annotate(i, xy=(frame.loc[i, "Std_Risk"] + 0.001, frame.loc[i, "Mean_Reward"] + 0.001), size = 15)
        plt.ylim(0, 0.3)
        plt.grid(True)
        plt.xlabel("Risk (Std)", fontsize = 15)
        plt.ylabel("Return (Mean)", fontsize = 15)
        plt.title("Annualized Risk & Return", fontsize = 20)
        plt.show()

    