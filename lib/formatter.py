import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pytickersymbols import PyTickerSymbols

stock_data = PyTickerSymbols()
listed_indices = stock_data.get_all_indices()


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
    ticker_list = []
    ticker_str = ""
    if not isinstance(tickers, list) and not isinstance(tickers, str):
        return "Invalid ticker input."
    if isinstance(tickers, list):
        ticker_str = (", ".join(tickers)).upper()
        ticker_list = tickers
    elif isinstance(tickers, str):
        ticker_list = re.findall("[a-zA-Z^]+", tickers.upper())
        ticker_str = ", ".join(ticker_list)
    return ticker_list, ticker_str


def valid_tickers(ticker_list,text_out=1):
    new_list = []
    for ticker in ticker_list:
        yf_ticker = yf.Ticker(ticker)

        # Slower method
        # yf_ticker.history(start="2024-06-01", end="2024-06-15")
        # error_message = yf.shared._ERRORS[ticker.upper()]
        # if "delisted" in error_message:
        #     print(f"{ticker} not a valid ticker.")

        info = yf_ticker.info
        if len(info) == 1:
            print(f"{ticker} not a valid ticker.")
        else:
            new_list.append(ticker)
    if len(new_list) == 0:
        return "No valid tickers."
    else:
        if text_out==1:
            print("Only keeping valid tickers.")
        return new_list


def yes_no(question="Y/N"):
    while True:
        user_input = input(question)
        if user_input.lower()[0] == "y":
            answ = 1
            break
        elif user_input.lower()[0] == "n":
            answ = 0
            break
        else:
            print("Invalid input. Please enter yes/no.")
    return answ


class DataToFormat:
    def __init__(self, tickers, start_date, end_date):
        self.ticker_list, self.ticker_str = ticker_formatter(tickers)
        self.ticker_list = valid_tickers(self.ticker_list)
        self.ticker_str = ", ".join(self.ticker_list)
        print(self.ticker_list)
        self.start_date = date_formatter(start_date)
        self.end_date = date_formatter(end_date)

    def __str__(self):
        return f"{self.ticker_str} data between {self.start_date} and {self.end_date}"

    def index_stocks(self):
        global listed_indices
        indices = [ticker for ticker in self.ticker_list if "^" in ticker]
        index_stock_dict = {}
        for index in indices:
            question = f"Include data for all stocks in {index} index?"
            include = yes_no(question)
            if include == 1:
                index_tickers = []
                yf_ticker = yf.Ticker(index)
                info = yf_ticker.info
                info_keys = info.keys()
                if "shortName" not in info_keys and "longName" not in info_keys:
                    print(f"Cannot find stocks in {index}.")
                elif "shortName" in info_keys and info["shortName"] in listed_indices:
                    s_name = yf_ticker.info["shortName"]
                    index_stock_data = stock_data.get_stocks_by_index(s_name)
                    for company in index_stock_data:
                        symbol_list = company['symbols']
                        symb = company['symbol']
                        valid_tick = valid_tickers([symb],0)
                        if len(valid_tick) != 0:
                            index_tickers.append(valid_tick[0])
                        elif len(valid_tick) == 0 and len(symbol_list) != 0:
                            for inner_dict in symbol_list:
                                valid_tick = valid_tickers(inner_dict['yahoo'],0)
                                if len(valid_tick) != 0:
                                    index_tickers.append(valid_tick)
                                    break
                    index_stock_dict[index] = index_tickers
                elif "longName" in info_keys and info["longName"] in listed_indices:
                    l_name = yf_ticker.info["longName"]
                    index_stock_data = stock_data.get_stocks_by_index(l_name)
                    index_tickers = [stock['symbol'] for stock in index_stock_data]
                    index_tickers = valid_tickers(index_tickers)
                    index_stock_dict[index] = index_tickers
                else:
                    print(f"Cannot find stocks in {index}.")

        return index_stock_dict

    def pandas_data(self):
        index_stock_dic = self.index_stocks()

        if index_stock_dic == {}:
            data = yf.download(self.ticker_str, start=self.start_date, end=self.end_date, group_by='tickers')
        else:
            total_tickers = []
            for index in index_stock_dic.keys():
                total_tickers.append(index)
                total_tickers = total_tickers + index_stock_dic[index]
            missing_tickers = [ticker for ticker in self.ticker_list if ticker not in total_tickers]
            total_tickers = total_tickers + missing_tickers
            total_tickers_str = ", ".join(total_tickers).upper()
            data = yf.download(total_tickers_str, start=self.start_date, end=self.end_date, group_by='tickers')
        return data

    def csv_output(self):
        data = self.pandas_data()
        namel = self.ticker_list + [self.start_date, self.end_date]
        name = '_'.join(namel) + ".csv"
        return data.to_csv(name)


def user_input():
    ticker_list, ticker_str = ticker_formatter(input("Input tickers:"))
    print(f"Inputted tickers: {ticker_list}")
    ticker_list = valid_tickers(ticker_list)
    print(f"Valid tickers: {ticker_list}")
    while True:
        start_date = date_formatter(input("Enter start date:"))
        if "-" in start_date:
            break
        else:
            print("Invalid.")
    while True:
        end_date = date_formatter(input("Enter end date:"))
        if "-" in end_date:
            break
        else:
            print("Invalid.")
    return ticker_str, start_date, end_date


index_stock_data = stock_data.get_stocks_by_index("S&P 500")
# print([_["symbols"] for _ in index_stock_data])

j = DataToFormat("^GSPC, AMZN", "2020-01-01", "2020-03-01")
j.csv_output()

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
