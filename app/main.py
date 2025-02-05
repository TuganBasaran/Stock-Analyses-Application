from InstrunmentComparitor import InstrunmentComparitor as IC

tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
start = "2015-01-01" 
end =  "2019-12-31"
comparitor = IC(tickers = tickers, start = start, end = end)
comparitor.print_head_ins()