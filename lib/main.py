import pandas as pd
import yfinance

from formatter import *

from scipy.interpolate import PchipInterpolator


def main():
    # j = DataToFormat("^DJI", "2020-01-01", "2020-01-14")
    # df=j.df_cleaning(j.pandas_index_components)
    # j.excel_output(j.pandas_index_components)
    # csv_path = 'AAPL_AMZN_2020-01-01_2020-01-14_pandas_tickers.csv'
    # df=csv_input(csv_path)

    apl = Option('AAPL')
    # a, b, c = apl.arbitrage_conditions()
    apl.interp_noArb(method=2).to_excel('AAPL_interp_constraints.xlsx')
    # print(d)
    # apl.arbitrage_conditions().to_excel('AAPL_options.xlsx')

    pass


if __name__ == "__main__":
    main()
