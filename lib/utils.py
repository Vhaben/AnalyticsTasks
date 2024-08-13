import re
import requests

import pandas as pd
import numpy as np

def yes_no(question="Y/N?"):
    while True:
        answer = input(question)
        if answer.lower()[0] == "y":
            ans = 1
            break
        elif answer.lower()[0] == "n":
            ans = 0
            break
        else:
            print("Invalid input. Please enter Y or N.")
    return ans


def user_input():
    ticker_list, ticker_str = ticker_formatter(input("Input tickers:"))
    print(f"Inputted tickers: {ticker_list}")
    # Can remove ticker validation
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

def csv_input(csv_path, index_stocks=0):
    # Read the CSV file. The file has multi-level headers, hence header=[0, 1].
    if index_stocks == 0:
        # If CSV was created to store ticker info only
        header_index = [0, 1]
        header_tuple = ('Unnamed: 0_level_0', 'Unnamed: 0_level_1')
    elif index_stocks == 1:
        # If CSV was created to store info of all stocks in a ticker
        header_index = [0, 1, 2]
        header_tuple = ('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'Unnamed: 0_level_2')
    else:
        print("Second argument must be 0 or 1.")
        return None
    df = pd.read_csv(csv_path, header=header_index)

    # Drop the first row as it contains only the Date information in one column, which is redundant after setting the index.
    df.drop(index=0, inplace=True)

    # Convert the 'Unnamed: 0_level_0', 'Unnamed: 0_level_1' column (which represents dates) to datetime format.
    # This assumes the dates are in the 'YYYY-MM-DD' format.
    df[header_tuple] = pd.to_datetime(df[header_tuple])

    # Set the datetime column as the index of the DataFrame. This makes time series analysis more straightforward.
    df.set_index(header_tuple, inplace=True)

    # Clear the name of the index to avoid confusion, as it previously referred to the multi-level column names.
    df.index.name = None
    return df