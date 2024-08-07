import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pytickersymbols import PyTickerSymbols
from scipy import interpolate

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


def valid_tickers(ticker_list, text_out=1):
    new_list = []
    invalid = []
    for ticker in ticker_list:
        yf_ticker = yf.Ticker(ticker)

        # Slower method
        # yf_ticker.history(start="2024-06-01", end="2024-06-15")
        # error_message = yf.shared._ERRORS[ticker.upper()]
        # if "delisted" in error_message:
        #     print(f"{ticker} not a valid ticker.")

        info = yf_ticker.info
        if len(info) == 1:
            if text_out == 1:
                print(f"{ticker} not a valid ticker.")
            invalid.append(ticker)
        else:
            new_list.append(ticker)
    if text_out == 1:
        if len(new_list) == 0:
            print("No valid tickers.")
        else:
            print("Only keeping valid tickers.")
    return new_list, invalid


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


def stocks_in_index(index, info_name):
    index_tickers = []
    yf_ticker = yf.Ticker(index)
    name = yf_ticker.info[info_name]
    index_stock_data = stock_data.get_stocks_by_index(name)
    lost_tickers = []
    for company in index_stock_data:
        symbol_list = company['symbols']
        symb = company['symbol']
        if symb is not None:
            valid_tick, lost = valid_tickers([symb], 0)
            if len(valid_tick) != 0:
                index_tickers.append(valid_tick[0])  # 'symbol' is valid for YFinance
        elif len(symbol_list) != 0 and (symb is None or len(valid_tick) == 0):  # 'symbol' not valid, so check 'symbols'
            for inner_dict in symbol_list:
                valid_tick = valid_tickers([inner_dict['yahoo']], 0)[0]
                if len(valid_tick) != 0:  # Found valid ticker in 'symbols'
                    index_tickers.append(valid_tick[0])
                    break
        if symb is not None and len(valid_tick) == 0:  # If didn't find any valid YF symbol for this stock
            lost_tickers.append(lost[0])
            print(f"{lost[0]} not found.")
    return index_tickers, lost_tickers


def unpack_index(index):
    global listed_indices
    yf_ticker = yf.Ticker(index)
    info = yf_ticker.info
    info_keys = info.keys()
    if "shortName" not in info_keys and "longName" not in info_keys:
        print(f"Cannot find stocks in {index}.")
        return None, None
    elif "shortName" in info_keys and info["shortName"] in listed_indices:
        return stocks_in_index(index, "shortName")
    elif "longName" in info_keys and info["longName"] in listed_indices:
        return stocks_in_index(index, "longName")
    else:
        print(f"Cannot find stocks in {index}.")
        return None, None


def df_cleaning(df):
    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(how='all', axis=0, inplace=True)
    # empty_cells = list(zip(np.where(pd.isnull(df))))
    df.interpolate(method='cubicspline')
    tickers = df.index.unique()
    for ticker in tickers:
        # wrong_highs=np.where(df[ticker]["High"] <= max(df[ticker]["Open"], df[ticker]["Close"]))
        # wrong_lows=np.where(df[ticker]["Low"] >= max(df[ticker]["Open"], df[ticker]["Close"]))
        df.loc[df[ticker]["High"] <= max(df[ticker]["Open"], df[ticker]["Close"]),"High"] =np.nan
        df.loc[df[ticker]["Low"] >= max(df[ticker]["Open"], df[ticker]["Close"]),"Low"] =np.nan
    df.interpolate(method='cubicspline')
    return df

class DataToFormat:
    def __init__(self, tickers, start_date, end_date):
        self.ticker_list, self.ticker_str = ticker_formatter(tickers)
        self.ticker_list = valid_tickers(self.ticker_list)[0]
        self.ticker_str = ", ".join(self.ticker_list)
        print(self.ticker_list)
        self.start_date = date_formatter(start_date)
        self.end_date = date_formatter(end_date)
        self.indices = [ticker for ticker in self.ticker_list if "^" in ticker]

    def __str__(self):
        return f"{self.ticker_str} data between {self.start_date} and {self.end_date}"

    def pandas_tickers(self):  # Return data of inputted tickers only
        data = yf.download(self.ticker_str, start=self.start_date, end=self.end_date, group_by='tickers')
        return data

    def pandas_index_stocks(self):  # Return data of stocks in indices entered only
        total_tickers = {}
        all_data = pd.DataFrame()
        for index in self.indices:
            index_tickers, lost_tickers = unpack_index(index)
            if index_tickers is not None and len(index_tickers) != 0:
                total_tickers[index] = index_tickers
                tickers_str = ", ".join(index_tickers)
                data = yf.download(tickers_str, start=self.start_date, end=self.end_date, group_by='tickers')
                data.columns = pd.MultiIndex.from_tuples([(index,) + _ for _ in data.columns])
                all_data = pd.concat([all_data, data], axis=1)
        if not all_data.empty:
            return all_data

    def csv_output(self, func):
        data = func()
        namel = self.ticker_list + [self.start_date, self.end_date]
        name = '_'.join(namel) + ".csv"
        print(data)
        if data is not None and not data.empty:
            return data.to_csv(name)


def user_input():
    ticker_list, ticker_str = ticker_formatter(input("Input tickers:"))
    print(f"Inputted tickers: {ticker_list}")
    ticker_list = valid_tickers(ticker_list)[0]
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


def csv_input(csv):
    return pd.read_csv(csv)


index_stock_data = stock_data.get_stocks_by_index("S&P 500")
# print([_["symbols"] for _ in index_stock_data])

j = DataToFormat("^GSPC", "2020-01-01", "2020-01-15")
j.csv_output(j.pandas_index_stocks)


class Index:
    pass


class Stock:
    pass

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
