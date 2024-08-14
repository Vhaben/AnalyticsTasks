import pandas as pd
import yfinance

from formatter import *


def main():
    # j = DataToFormat("AAPL AMZN", "2020-01-01", "2020-01-14")
    # df=j.df_cleaning(j.pandas_tickers)
    # j.excel_output(j.pandas_tickers)
    # csv_path = '^FCHI_2020-01-01_2020-01-14_pandas_index_stocks.csv'
    # csv_path = 'AAPL_AMZN_2020-01-01_2020-01-14_pandas_tickers.csv'
    # df=csv_input(csv_path)

    apl = Option('AAPL')
    # Expiration dates
    print(apl.risk_free_rate())
    # print(apl.options_expiration())
    apl.options_expiration().to_excel('AAPL_options.xlsx')
    pass


if __name__ == "__main__":
    main()

