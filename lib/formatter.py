import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import re
import datetime
class DataToFormat:
    def __init__(self,tickers,start_date,end_date):
        if isinstance(tickers,list):
            ticker_str = (", ".join(tickers)).upper()
        elif isinstance(tickers, str):
            ticker_list=re.findall("[a-zA-Z^]+", tickers.upper())
            ticker_str = ", ".join(ticker_list)
        self.ticker_str=ticker_str
        self.ticker_list=ticker_list
        try:
            self.start_date=datetime.datetime.strptime(start_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        except:
            try:
                self.start_date = datetime.datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
            except:
                self.start_date=start_date
        try:
            self.end_date=datetime.datetime.strptime(end_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        except:
            try:
                self.end_date = datetime.datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
            except:
                self.end_date=end_date

    def __str__(self):
        return f"{self.ticker_str} data between {self.start_date} and {self.end_date}"

    def pandas_data(self):
        # if len(self.ticker_list)==1:
            # Not using yf.downloads() module as not downloading many tickers' data at once
            # tickername=yf.Ticker(self.ticker_str)
        data = yf.download(self.ticker_str, start=self.start_date, end=self.end_date,group_by='tickers')
        # data['Date']=pd.to_datetime(data.index)
        return data

    def csv_output(self):
        data=self.pandas_data()
        namel=self.ticker_list+[self.start_date,self.end_date]
        name='_'.join(namel)+".csv"
        return data.to_csv(name)

j=DataToFormat("^GSPC amzn","19/06/2002","19-06-2003")
print(j)
print(j.pandas_data())
j.csv_output()