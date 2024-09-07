from formatter_index_stock import *


def main():
    j = DataToFormat("^DJI", "2020-01-01", "2020-01-14")
    # print(j.pandas_tickers())
    # df = j.df_cleaning(j.pandas_index_components)
    # print(df)

    # j.excel_output(j.pandas_index_components)
    # csv_path = 'AAPL_AMZN_2020-01-01_2020-01-14_pandas_tickers.csv'
    # df=csv_input(csv_path)
    pass


if __name__ == "__main__":
    main()
