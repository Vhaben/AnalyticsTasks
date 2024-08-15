import pandas as pd
import yahooquery as yq
from pytickersymbols import PyTickerSymbols
from scipy import interpolate

import logging

logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False

from utils import *

stock_data = PyTickerSymbols()
listed_indices = stock_data.get_all_indices()


class YfInput:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.yf_ticker = yf.Ticker(self.symbol)

    def valid_ticker(self, text_out=1):
        yf_ticker = self.yf_ticker
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
        global index_components
        if self.symbol in index_components:
            index_info = index_components[self.symbol]
            if isinstance(index_info, tuple):
                tickers = pd.read_html(
                    index_info[0])[index_info[1]]
                ticklist = tickers.Symbol.tolist()
                return [Stock(_) for _ in ticklist]
            elif isinstance(index_info, list):
                return [Stock(_) for _ in index_info]
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
            # print("-------------------------------------")
            # print(symbol)
            # print(symbol_list)
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
        # print("Lost:", lost_tickers)
        # print("Found:", index_tickers)
        return index_tickers, lost_tickers

    def unpack_index(self):
        wiki_tickers = self.yf_index_ticker_finder()
        if isinstance(wiki_tickers, list):
            return wiki_tickers, []
        else:
            global listed_indices
            yf_ticker = self.yf_ticker
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


class Option(YfInput):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.expiration_dates = self.yf_ticker.options
        try:
            self.currency = self.yf_ticker.info['currency']
        except:
            self.currency = None
        self.times_maturity=[]
        try:
            self.underlying = self.yf_ticker.history(start= (datetime.datetime.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')).Close.iloc[0]
        except:
            self.underlying = None

    def options_expiration(self):
        if len(self.expiration_dates) == 0:
            return None
        else:
            df_list = []
            for exp_date in self.expiration_dates:
                opt_chain = self.yf_ticker.option_chain(exp_date)
                calls = opt_chain.calls
                puts = opt_chain.puts
                calls.set_index('strike', inplace=True)
                puts.set_index('strike', inplace=True)

                # ! Check calculation
                time_to_mat = (datetime.datetime.strptime(exp_date,
                                                          "%Y-%m-%d") - datetime.datetime.today()).total_seconds() / (
                                          24 * 60 * 60) / 365

                self.times_maturity.append(time_to_mat)

                # Add optionType column
                # calls['optionType'] = 'C'
                # puts['optionType'] = 'P'

                # Merge calls and puts into a single dataframe
                options = pd.concat(objs=[calls, puts], axis=1, keys=['Call', 'Put'], )
                options.drop(
                    columns=list(itertools.product(['Call', 'Put'],
                                                   ['contractSize', 'currency', 'change', 'percentChange',
                                                    'lastTradeDate',
                                                    'lastPrice'])),
                    inplace=True)
                options['expirationDate'] = exp_date
                options['expirationDate'] = pd.to_datetime(options['expirationDate'])

                # ! Check calculation
                options['timeToMaturity'] = (options[
                                                 'expirationDate'] - pd.Timestamp.today().normalize()).dt.days / 365
                options.columns = pd.MultiIndex.from_tuples([(time_to_mat,) + _ for _ in options.columns])
                df_list.append(options)
            options_df = pd.concat(df_list, axis=1)
            options_df.sort_index(inplace=True)
            return options_df

    def risk_free_rate(self):
        global risk_free_rate_sources
        if self.currency is not None:
            if self.currency in risk_free_rate_sources:
                rate_info=risk_free_rate_sources[self.currency]
                if isinstance(rate_info,str):
                    rfr = pdr.DataReader(rate_info, "fred").iloc[-1,0]
                    self.rfr=rfr
                    return rfr


    def arbitrage_conditions(self):
        self.risk_free_rate()
        df=self.options_expiration()
        df = df.dropna(how='all', axis=1)
        df = df.dropna(how='all', axis=0)
        for time_to in self.times_maturity:
            # print(df[time_to])
            df[time_to,'parity','']=self.underlying - df.index * np.exp(-self.rfr * time_to)

            # ! Change 'price' to correct column
            # df.loc[max(0,df[time_to, 'parity','']) > df[time_to,'Call','price'] | df[time_to,'Call','price']> self.underlying] = np.nan
            # df.loc[min(0, df[time_to, 'parity', '']) > df[time_to, 'Put', 'price'] | df[
                # time_to, 'Put', 'price'] > - df[time_to,'parity',''] + self.underlying] = np.nan

            # First partial derivatives

            # Second partial derivatives
        return df


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
        return f'{self.ticker_str} data between {self.start_date} and {self.end_date}'

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

    def pandas_index_components(self):
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

    def df_cleaning(self, func):
        # removed_stocks=[col for col in df.columns if df[col].isna().all()])
        df = func()
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

    def csv_output(self, func, cleaning=1):
        if cleaning == 0:
            data = func()
        else:
            data = self.df_cleaning(func)
        print(data)
        if data is not None:
            namel = self.ticker_list + [self.start_date, self.end_date]
            name = '_'.join(namel) + "_" + func.__name__ + ".csv"
            if not data.empty:
                return data.to_csv(name)

    def excel_output(self, func, cleaning=1):
        if cleaning == 0:
            data = func()
        else:
            data = self.df_cleaning(func)
        if data is not None:
            namel = self.ticker_list + [self.start_date, self.end_date]
            name = '_'.join(namel) + "_" + func.__name__ + ".xlsx"
            if not data.empty:
                return data.to_excel(name)
