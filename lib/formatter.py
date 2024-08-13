import re
import requests

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import yahooquery as yq
from pytickersymbols import PyTickerSymbols
from scipy import interpolate

import logging

logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False

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


def df_cleaning(df):
    # removed_stocks=[col for col in df.columns if df[col].isna().all()])
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)
    # empty_cells = list(zip(np.where(pd.isnull(df))))
    df.astype(float).interpolate(method='cubicspline')
    # ticker_level=len(df.columns.levels)-2 # Level of stock tickers in multi-index columns
    tickers = pd.MultiIndex.from_tuples([tuple(col[:-1]) for col in df.columns]).unique()
    # print(tickers)
    for ticker in tickers:
        df.loc[df[ticker + ("High",)] <= df[[ticker + ("Open",), ticker + ("Close",)]].max(axis=1), ticker + (
            "High",)] = np.nan
        df.loc[df[ticker + ("Low",)] >= df[[ticker + ("Open",), ticker + ("Close",)]].max(axis=1), ticker + (
            "Low",)] = np.nan
    df.astype(float).interpolate(method='cubicspline')
    return df


class YfInput:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def valid_ticker(self, text_out=1):
        yf_ticker = yf.Ticker(self.symbol)
        # Slower method
        # yf_ticker.history(start="2024-06-01", end="2024-06-15")
        # error_message = yf.shared._ERRORS[ticker.upper()]
        # if "delisted" in error_message:
        #     print(f"{ticker} not a valid ticker.")
        info = yf_ticker.info
        if len(info) == 1:
            if text_out == 1:
                print(f"{self.symbol} not a valid ticker.")
            return False
        else:
            if text_out == 1:
                print(f"{self.symbol} valid.")
            return True


class Index(YfInput):
    def yf_index_ticker_finder(self):
        # Returns:
        # - List of Stock objects
        # - Symbol of index
        if self.symbol == '^GSPC':
            tickers = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            ticklist = tickers.Symbol.tolist()
            return [Stock(_) for _ in ticklist]
        elif self.symbol == '^DJI':
            tickers = pd.read_html(
                'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
            ticklist = tickers.Symbol.tolist()
            return [Stock(_) for _ in ticklist]
        elif self.symbol == '^OEX':
            tickers = pd.read_html(
                'https://en.wikipedia.org/wiki/S%26P_100')[2]
            ticklist = tickers.Symbol.tolist()
            return [Stock(_) for _ in ticklist]
        elif self.symbol == '^FCHI':
            # df = pd.read_html('https://www.bnains.org/archives/action.php/')[0]['Code ISIN'].squeeze()
            # ISIN_list= df.to_list()
            #
            # tick_list=[yq.search(_)["quotes"][0]["symbol"] for _ in ISIN_list]
            # vtickers = pd.read_html('https://fr.wikipedia.org/wiki/CAC_40')[2].iloc[:, 1].to_list()
            cac40_tickers = ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA',
                             'ACA.PA', 'BN.PA',
                             'DSY.PA', 'EDEN.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA',
                             'MC.PA',
                             'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA',
                             'GLE.PA',
                             'STLAP.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.PA', 'VIE.PA', 'DG.PA', 'VIV.PA']
            return [Stock(_) for _ in cac40_tickers]
        else:
            return self.symbol

    def stocks_in_index(self, info_name, text_out=1):
        # Returns:
        # - index_tickers: list of Stock objects
        # - lost_tickers: list of strings (invalid tickers)
        global stock_data
        index_tickers = []
        yf_ticker = yf.Ticker(self.symbol)
        name = yf_ticker.info[info_name]
        index_stock_data = stock_data.get_stocks_by_index(name)
        lost_tickers = []
        for company in index_stock_data:
            symbol_list = company['symbols']
            symbol = company['symbol']
            valid = False
            print("-------------------------------------")
            print(symbol)
            print(symbol_list)
            if symbol is not None:
                stock_symb = Stock(company['symbol'])
                valid = stock_symb.valid_ticker(0)
                if valid:
                    index_tickers.append(stock_symb)  # 'symbol' is valid for YFinance
            if len(symbol_list) != 0 and (
                    symbol is None or not valid):  # 'symbol' not valid, so check 'symbols'
                for inner_dict in symbol_list:
                    stock_symb = Stock(inner_dict['yahoo'])
                    print("!", stock_symb)
                    valid = stock_symb.valid_ticker(0)
                    if valid:  # Found valid ticker in 'symbols'
                        index_tickers.append(stock_symb)
                        break
            if not valid:  # If didn't find any valid YF symbol for this stock
                if symbol is not None:
                    lost_tickers.append(symbol)
                    if text_out == 1:
                        print(f"{symbol} not found.")
                else:
                    lost_tickers.append(company)
        print("Lost:", lost_tickers)
        print("Found:", index_tickers)
        return index_tickers, lost_tickers

    def unpack_index(self):
        wiki_tickers = self.yf_index_ticker_finder()
        if isinstance(wiki_tickers, list):
            return wiki_tickers, []
        else:
            global listed_indices
            yf_ticker = yf.Ticker(self.symbol)
            info = yf_ticker.info
            info_keys = info.keys()
            if "shortName" in info_keys and info["shortName"] in listed_indices:
                return self.stocks_in_index("shortName", 0)
            elif "longName" in info_keys and info["longName"] in listed_indices:
                return self.stocks_in_index("longName", 0)
            else:
                # matches=[_ for _ in listed_indices if re.findall("[a-zA-Z^]+", info["shortName"].upper())[0] in _]
                # if len(matches)!=0:
                #     # Find index of closest matching name in PytickerSymbols database
                #     return self.stocks_in_index(matches[0], 0)
                print(f"Cannot find stocks in {self.symbol}.")
                return None, None


class Stock(YfInput):
    pass


class DataToFormat:
    def __init__(self, tickers, start_date="", end_date=""):
        self.ticker_list, self.ticker_str = ticker_formatter(tickers)
        # Ticker validation
        # self.ticker_list = valid_tickers(self.ticker_list)[0]
        self.ticker_str = ", ".join(self.ticker_list)
        self.start_date = date_formatter(start_date)
        self.end_date = date_formatter(end_date)
        self.ticker_types = [Index(_) if "^" in _ else Stock(_) for _ in self.ticker_list]
        self.indices = [_ for _ in self.ticker_types if isinstance(_, Index)]

    def __str__(self):
        return f"{self.ticker_str} data between {self.start_date} and {self.end_date}"

    def pandas_tickers(self, text_out=1):
        # Return data of inputted tickers only
        failed_tickers = []
        all_data = pd.DataFrame()
        for ticker in self.ticker_types:
            data = yf.download(ticker.symbol, start=self.start_date, end=self.end_date, group_by='tickers')
            if data.empty:
                failed_tickers.append(ticker)
                if text_out == 1:
                    print(f"Ticker {ticker.symbol} failed.")
            else:
                data.columns = pd.MultiIndex.from_tuples([(ticker.symbol, _) for _ in data.columns])
                all_data = pd.concat([all_data, data], axis=1)
        # data = yf.download(self.ticker_str, start=self.start_date, end=self.end_date, group_by='tickers')
        return all_data

    def pandas_index_stocks(self):
        # Return data of stocks in indices entered only
        total_tickers = {}
        all_data = pd.DataFrame()
        for index in self.indices:
            index_tickers, lost_tickers = index.unpack_index()
            if index_tickers is not None and len(index_tickers) != 0:
                total_tickers[index.symbol] = index_tickers
                tickers_str = ", ".join([_.symbol for _ in index_tickers])
                # print(tickers_str)
                data = yf.download(tickers_str, start=self.start_date, end=self.end_date, group_by='tickers')
                data.columns = pd.MultiIndex.from_tuples([(index.symbol,) + _ for _ in data.columns])
                all_data = pd.concat([all_data, data], axis=1)
        if not all_data.empty:
            return all_data

    def csv_output(self, func):
        data = func()
        print(data)
        if data is not None:
            namel = self.ticker_list + [self.start_date, self.end_date]
            name = '_'.join(namel) + func.__name__ + ".csv"
            if not data.empty:
                return data.to_csv(name)

    def excel_output(self, func):
        data = func()
        if data is not None:
            namel = self.ticker_list + [self.start_date, self.end_date]
            name = '_'.join(namel) + func.__name__ + ".xlsx"
            if not data.empty:
                return data.to_excel(name)

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
