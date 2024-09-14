from utils import *


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
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.stocks_of_index = None
        self.lost_tickers = None

    def yf_index_ticker_finder(self):
        # Retrieve data from index_components global dict
        # Returns:
        # - List of Stock objects
        # - Symbol of index

        global index_components
        if self.symbol in index_components:
            index_info = index_components[self.symbol]
            if isinstance(index_info, tuple):
                tickers = pd.read_html(
                    index_info[0])[index_info[1]]  # First element of tuple is link, second is index of table at the URL
                ticklist = tickers.Symbol.tolist()
                return [Stock(_) for _ in ticklist]
            elif isinstance(index_info, list):
                return [Stock(_) for _ in index_info]  # List of stock tickers
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
        # If unpack_index function already run, stocks_of_index already calculated, so ask if want to rerun
        if already_run("unpack_index", self.stocks_of_index):

            # Retrieve stock tickers preferentially from global index_components dict
            global index_components
            if self.symbol in index_components:
                self.stocks_of_index = self.yf_index_ticker_finder()
                return self.stocks_of_index, []  # Empty list as no lost tickers

            # Otherwise retrieve tickers from PyTickerSymbols package
            else:
                global listed_indices
                yf_ticker = self.yf_ticker
                info = yf_ticker.info
                info_keys = info.keys()
                if "shortName" in info_keys and info["shortName"] in listed_indices:
                    self.stocks_of_index, self.lost_tickers = self.stocks_in_index("shortName", 0)
                    return self.stocks_of_index, self.lost_tickers
                elif "longName" in info_keys and info["longName"] in listed_indices:
                    self.stocks_of_index, self.lost_tickers = self.stocks_in_index("longName", 0)
                    return self.stocks_of_index, self.lost_tickers
                else:
                    # matches=[_ for _ in listed_indices if re.findall("[a-zA-Z^]+", info["shortName"].upper())[0] in _]
                    # if len(matches)!=0:
                    #     # Find index of closest matching name in PytickerSymbols database
                    #     return self.stocks_in_index(matches[0], 0)
                    print(f"Cannot find stocks in {self.symbol}.")
                    return None, None
        else:
            return self.stocks_of_index, self.lost_tickers


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
        self.ticker_data = None
        self.index_components_data = None
        self.interped_data = None

        # Map for functions for Excel output
        self.func_output_map: {DataToFormat.df_cleaning: self.interped_data,
                               DataToFormat.pandas_tickers: self.ticker_data,
                               DataToFormat.pandas_index_components: self.index_components_data}

    def __str__(self):
        return f'{self.ticker_str} data between {self.start_date} and {self.end_date}'

    def pandas_tickers(self, text_out=1):
        # Return data of inputted tickers only

        if already_run("pandas_tickers", self.ticker_data):
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
            self.ticker_data = all_data

        return self.ticker_data

    def pandas_index_components(self, run=0):
        # Return data of stocks in indices entered only

        if run == 1 or already_run("pandas_index_components", self.index_components_data):
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
                self.index_components_data = all_data
        self.index_components_data.index = self.index_components_data.index.tz_localize(None)
        return self.index_components_data

    def df_cleaning(self, func, run=0):
        # removed_stocks=[col for col in df.columns if df[col].isna().all()])

        if run == 1 or already_run("df_cleaning", self.interped_data):
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
            self.interped_data = df
        return self.interped_data

    def csv_output(self, func, cleaning=1):
        if cleaning == 0:
            asked_output = self.func_output_map[func]
            if already_run(func, asked_output):
                data = func(run=1)
            else:
                data = asked_output
            name = '_'.join(self.ticker_list + [self.start_date, self.end_date]) + "_" + func.__name__ + ".csv"
        else:
            if already_run(func, self.interped_data):
                data = self.df_cleaning(func, run=1)
            else:
                data = self.interped_data
            name = '_'.join(
                self.ticker_list + [self.start_date, self.end_date, 'interped']) + "_" + func.__name__ + ".csv"
        # print(data)
        if data is not None and not data.empty:
            return data.to_csv(name)

    def excel_output(self, func, cleaning=1):
        if cleaning == 0:
            asked_output = self.func_output_map[func]
            if already_run(func, asked_output):
                data = func(run=1)
            else:
                data = asked_output
            name = '_'.join(self.ticker_list + [self.start_date, self.end_date]) + "_" + func.__name__ + ".xlsx"
        else:
            if already_run(func, self.interped_data):
                data = self.df_cleaning(func, run=1)
            else:
                data = self.interped_data
            name = '_'.join(
                self.ticker_list + [self.start_date, self.end_date, 'interped']) + "_" + func.__name__ + ".xlsx"
        if data is not None and not data.empty:
            return data.to_excel(name)
