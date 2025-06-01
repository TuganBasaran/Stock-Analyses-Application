from InstrunmentComparitor import InstrunmentComparitor as IC
from FinancialInstrunment import FinancialInstrunment as FI 

tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
ticker = ['FROTO.IS']
start = "2015-01-01" 
end =  "2019-12-31"


# ticker = input("Please enter ticker from Yahoo Finace: ")
# start = input("Please enter start date like: (2019-12-31): ")
# end = input("Please enter your end date like before: ")

# instrument = FI(ticker, start, end) 
instrunment_2 = IC(tickers, start, end)
instrunment_2.get_data()