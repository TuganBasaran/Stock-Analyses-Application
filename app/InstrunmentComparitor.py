import pandas as pd 
import numpy as np 
import yfinance as yf
import scipy.stats as stats
import matplotlib.pyplot as plt
from FinancialInstrunment import FinancialInstrunment

'''
TODO: 
- Create correlation/covarience method 
- Create Reward/Risk scatter table method
'''

class InstrunmentComparitor(): 
    def __init__(self, tickers, start, end):
        self.tickers = tickers
        self.start = start 
        self.end = end 
        self.get_data()

    def __repr__(self):
        return f"Instrunment Comparitor (tickers = {self.tickers}, start={self.start}, end={self.end})"

    def get_data(self): 
        instrunments = []
        for i in self.tickers: 
            instrunment  = FinancialInstrunment(i, self.start, self.end)
            instrunments.append(instrunment)
        self.instrunments = instrunments

    def print_head_ins(self): 
        for i in self.instrunments: 
            print(i.data.head() , "\n")
        