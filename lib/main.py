import yfinance

from formatter import *
from utils import *


def main():
    # j = DataToFormat("AAPL AMZN", "2020-01-01", "2020-01-14")
    # j.excel_output(j.pandas_tickers)
    # csv_path = '^FCHI_2020-01-01_2020-01-14pandas_index_stocks.csv'
    csv_path = 'AAPL_AMZN_2020-01-01_2020-01-14pandas_tickers.csv'
    df=csv_input(csv_path)
    pass


if __name__ == "__main__":
    main()
