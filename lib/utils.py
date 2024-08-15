import re
import requests
import datetime
import itertools

import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

index_components = {'^GSPC': ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0),
                    '^DJI': ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 1),
                    '^OEX': ('https://en.wikipedia.org/wiki/S%26P_100', 2),
                    '^FCHI': ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA',
                              'ACA.PA', 'BN.PA',
                              'DSY.PA', 'EDEN.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA',
                              'MC.PA',
                              'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA',
                              'GLE.PA',
                              'STLAP.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.PA', 'VIE.PA', 'DG.PA',
                              'VIV.PA']}

# Alternative for CAC40
# df = pd.read_html('https://www.bnains.org/archives/action.php/')[0]['Code ISIN'].squeeze()
# ISIN_list= df.to_list()
#
# tick_list=[yq.search(_)["quotes"][0]["symbol"] for _ in ISIN_list]
# vtickers = pd.read_html('https://fr.wikipedia.org/wiki/CAC_40')[2].iloc[:, 1].to_list()

risk_free_rate_sources= { 'EUR' : 'IRLTLT01EZM156N',
                          'USD' : 'IRLTLT01USM156N'}

# Alternative for EUR
# 'EUR': ('https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html',0 ,(0,1))
# pd.read_html('https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html')[0].loc[0,1])

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
