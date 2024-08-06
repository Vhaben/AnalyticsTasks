import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pytickersymbols import PyTickerSymbols

global stock_data
stock_data = PyTickerSymbols()
global listed_indices
listed_indices= stock_data.get_all_indices()

def date_formatter(date):
    datelist = re.findall("[0-9]+", date)
    if len(datelist) != 3 or (len(datelist[0]) != 4 and len(datelist[2]) != 4):
        return "Invalid date format."
    if len(datelist[0]) == 4:
        date = "-".join(datelist)
    elif len(datelist[2]) == 4:
        date = datelist[2] + "-" + datelist[1] + "-" + datelist[0]
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except:
        return "Date numbers invalid."
    return date


def ticker_formatter(tickers):
    if not isinstance(tickers, list) and not isinstance(tickers, str):
        return "Invalid ticker input."
    if isinstance(tickers, list):
        ticker_str = (", ".join(tickers)).upper()
        ticker_list = tickers
    elif isinstance(tickers, str):
        ticker_list = re.findall("[a-zA-Z^]+", tickers.upper())
        ticker_str = ", ".join(ticker_list)
    return (ticker_list, ticker_str)


def valid_tickers(ticker_list):
    new_list=[]
    for ticker in ticker_list:
        yf_ticker = yf.Ticker(ticker)
        info = None

        # Slower method
        # yf_ticker.history(start="2024-06-01", end="2024-06-15")
        # error_message = yf.shared._ERRORS[ticker.upper()]
        # if "delisted" in error_message:
        #     print(f"{ticker} not a valid ticker.")

        info=yf_ticker.info;
        if len(info)==1:
            print(f"{ticker} not a valid ticker.")
        else:
            new_list.append(ticker)
    return new_list

def yes_no(question="Y/N"):
    answ=0
    while True:
        user_input = input(question)
        if user_input.lower()[0]=="y":
            answ=1
            break
        elif user_input.lower()[0]=="n":
            answ=0
            break
        else:
            print("Invalid input. Please enter yes/no.")
    return answ

class DataToFormat:
    def __init__(self, tickers, start_date, end_date):
        self.ticker_list, self.ticker_str = ticker_formatter(tickers)
        self.ticker_list=valid_tickers(self.ticker_list)
        self.start_date = date_formatter(start_date)
        self.end_date = date_formatter(end_date)

    def __str__(self):
        return f"{self.ticker_str} data between {self.start_date} and {self.end_date}"

    def index_stocks(self):
        indices = [ticker for ticker in self.ticker_list if "^" in ticker]
        index_stock_dict={}
        for index in indices:
            question=f"Include data for all stocks in {index} index?"
            include=yes_no(question)
            if include == 1:
                yf_ticker = yf.Ticker(index)
                info = yf_ticker.info
                info_keys = yf_ticker.info.keys()
                if "shortName" not in info_keys and "longName" not in info_keys:
                    print(f"Cannot find stocks in {index}.")
                elif "shortName" in info_keys and yf_ticker.info["shortName"] in listed_indices:
                    s_name = yf_ticker.info["shortName"]
                    index_stock_data = stock_data.get_stocks_by_index(s_name)
                    index_tickers = [stock['symbol'] for stock in index_stock_data]
                    index_stock_dict[index] =index_tickers
                elif "longName" in info_keys and yf_ticker.info["longName"] in listed_indices:
                    l_name = yf_ticker.info["longName"]
                    index_stock_data = stock_data.get_stocks_by_index(l_name)
                    index_tickers = [stock['symbol'] for stock in index_stock_data]
                    index_stock_dict[index] = index_tickers
                else:
                    print(f"Cannot find stocks in {index}.")
        return index_stock_dict


    def pandas_data(self):
        index_stock_dic=self.index_stocks()

        # if len(self.ticker_list)==1:
        # Not using yf.downloads() module as not downloading many tickers' data at once
        # tickername=yf.Ticker(self.ticker_str)


        data = yf.download(self.ticker_str, start=self.start_date, end=self.end_date, group_by='tickers')
        # data['Date']=pd.to_datetime(data.index)
        return data

    def csv_output(self):
        data = self.pandas_data()
        namel = self.ticker_list + [self.start_date, self.end_date]
        name = '_'.join(namel) + ".csv"
        return data.to_csv(name)


j = DataToFormat("^GSPC amzn", "19/06/2002", "19-06-2003")
j.index_stocks()



# nasdaq_tickers = stock_data.get_stocks_by_index('NASDAQ 100')  # Corrected index name

# Print the ticker symbols
# for stock in nasdaq_tickers:
#     print(stock['symbol'])
# all_ticker_getter_names = list(filter(
#    lambda x: (
#          x.endswith('_google_tickers') or x.endswith('_yahoo_tickers')
#    ),
#    dir(stock_data),
# ))
# print(all_ticker_getter_names)
# print(stock_data.get_dow_jones_london_yahoo_tickers())

#
# indices=[ticker for ticker in ticker_list if "^" in ticker]
#
# for index in indices:
#     yf_ticker=yf.Ticker(index)
#     info=yf_ticker.info
#     info_keys=yf_ticker.info.keys()
#     if "shortName" not in info_keys and "longName" not in info_keys:
#         print(f"Cannot find stocks in {index}.")
#     elif "shortName" in info_keys and yf_ticker.info["shortName"] in listed_indices:
#         s_name=yf_ticker.info["shortName"]
#         index_stock_data = stock_data.get_stocks_by_index(s_name)
#         index_tickers=[stock['symbol'] for stock in index_stock_data]
#     elif "longName" in info_keys and yf_ticker.info["longName"] in listed_indices:
#         l_name=yf_ticker.info["longName"]
#         index_stock_data = stock_data.get_stocks_by_index(s_name)
#         index_tickers = [stock['symbol'] for stock in index_stock_data]
#     else:
#         print(f"Cannot find stocks in {index}.")