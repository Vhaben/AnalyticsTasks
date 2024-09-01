import re
import requests
import datetime
import pytz
import itertools

import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate

import yfinance as yf
import yahooquery as yq
from pytickersymbols import PyTickerSymbols

import logging

logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False

stock_data = PyTickerSymbols()
listed_indices = stock_data.get_all_indices()

# Timezone offset from UTC for option time to maturity calculation (assuming end of day)
timezone_offset = {'EUR': 'Europe/Paris', 'USD': 'America/New_York'}

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

risk_free_rate_sources = {'EUR': (
    'https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html', 0,
    (0, 1)),
    'USD': 'SOFR'}


# Alternative for EUR
# 'EUR' :  'ECBESTRVOLWGTTRMDMNRT

def risk_free_rates():
    rates = {}
    global risk_free_rate_sources
    for currency in risk_free_rate_sources:
        currency_info = risk_free_rate_sources[currency]
        if isinstance(currency_info, str):
            rfr = pdr.DataReader(currency_info, "fred").iloc[-1, 0]
            rates[currency] = rfr
        elif isinstance(currency_info, tuple):
            rfr = pd.read_html(currency_info[0])[currency_info[1]].loc[currency_info[2]]
            rates[currency] = rfr
    return rates


rf_currencies = risk_free_rates()


def convexity_constraint(params, num_calls):
    # Convexity constraint for optimization after interpolation (not together) for Option.interp_noArb method
    return int(not (np.all(np.diff(params[:num_calls], n=2) > 0) and np.all(np.diff(params[num_calls:], n=2) > 0)))


def monoticity_constraint(params, num_calls):
    # Monotonicity constraint for optimization after interpolation (not together) for Option.interp_noArb method
    return int(not (np.all(np.diff(params[:num_calls], n=1) < 0) and np.all(np.diff(params[num_calls:], n=1) > 0)))


def parity_minimization(call_puts, spot_strike):
    # Objective function for minimization of put-call parity by mean squared error for Option.interp_noArb method
    # - call_puts: list of (interpolated) calls and puts
    # - spot_strike: list of put-call parity values for all strikes and times-to-maturity
    n = len(spot_strike)
    calls = call_puts[:n]
    puts = call_puts[n:]
    return np.mean((calls - puts - spot_strike) ** 2)


def parity_min_with_interp(trial, spot_strike, interped):
    # Objective function for minimization of put-call parity and mean squared error between interpolated values and new values for Option.interp_noArb method
    # - trial: list of (interpolated) calls and puts
    # - spot_strike: list of put-call parity values for all strikes and times-to-maturity
    # - interped: interpolated values of calls and puts
    return parity_minimization(trial, spot_strike) + np.mean((trial - interped) ** 2)


def spline_convexity(coefficients, n_call, all_strikes, all_times, valid_call_coordinates, valid_put_coordinates,
                     tck_call, tck_put):
    # Convexity constraint for iterative optimization of spline coefficients for Option.spline_interp method
    # - coefficients: coefficients of spline of call and put
    # - n_call: number of coefficients in coefficients list for call spline (rest are for put spline)
    # - all_strikes: list of all strikes (x-coordinate of points at which spline is evaluated)
    # - all_times: list of all times (y-coordinate of points at which spline is evaluated)
    # - valid_call_coordinates: 1D strike-time coordinates mask of calls that were not removed as not arbitrable
    # - valid_put_coordinates: 1D strike-time coordinates mask of puts that were not removed as not arbitrable
    # - tck_call: list output from interpolate.bisplrep with knots on calls; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    # - tck_put: list output from interpolate.bisplrep with knots on puts; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev

    # Current coefficients
    new_tck_calls = tck_call[0:2] + [coefficients[:n_call]] + tck_call[3:]
    new_tck_puts = tck_put[0:2] + [coefficients[n_call:]] + tck_put[3:]

    # Convexity as second-derivative along strike axis
    d2z_dx2_calls = interpolate.bisplev(all_strikes, all_times, new_tck_calls, dx=2, dy=0)
    d2z_dx2_puts = interpolate.bisplev(all_strikes, all_times, new_tck_puts, dx=2, dy=0)

    # Flatten along each column i.e. time-to-maturity first
    return min(np.min(np.ravel(d2z_dx2_calls,order='F')[valid_call_coordinates]), np.min(np.ravel(d2z_dx2_puts,order='F')[valid_put_coordinates]))


def bounds_vanilla_constraint(coefficients, n_call, all_strikes, all_times, call_bounds, put_bounds, tck_call, tck_put):
    # Convexity constraint for iterative optimization of spline coefficients for Option.spline_interp method
    # - coefficients: coefficients of spline of call and put
    # - n_call: number of coefficients in coefficients list for call spline (rest are for put spline)
    # - all_strikes: list of all strikes (x-coordinate of points at which spline is evaluated)
    # - all_times: list of all times (y-coordinate of points at which spline is evaluated)
    # - call_bounds: list of tuples of lower and upper bounds on call prices
    # - put_bounds: list of tuples of lower and upper bounds on put prices
    # - tck_call: list output from interpolate.bisplrep with knots on calls; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    # - tck_put: list output from interpolate.bisplrep with knots on puts; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev

    # Current coefficients
    new_tck_calls = tck_call[0:2] + [coefficients[:n_call]] + tck_call[3:]
    new_tck_puts = tck_put[0:2] + [coefficients[n_call:]] + tck_put[3:]

    # Newly calculated values at every grid point
    all_interp_calls = interpolate.bisplev(all_strikes, all_times, new_tck_calls)
    all_interp_puts = interpolate.bisplev(all_strikes, all_times, new_tck_puts)

    # Flatten along each column i.e. time-to-maturity first
    raveled_calls = all_interp_calls.ravel(order='F')
    raveled_puts = all_interp_puts.ravel(order='F')

    for strike_time in range(len(raveled_calls)):
        if ((not call_bounds[strike_time][0] < raveled_calls[strike_time] < call_bounds[strike_time][1])
                or (not put_bounds[strike_time][0] < raveled_puts[strike_time] < put_bounds[strike_time][1])):
            return -1
    return 1


def optimal_spline(coefficients, n_call, all_strikes, all_times, valid_call_coordinates, valid_put_coordinates,
        valid_call_prices, valid_put_prices, parity, tck_call, tck_put):
    # Objective function for iterative optimization of spline coefficients for Option.spline_interp method
    # - coefficients: coefficients of spline of call and put
    # - n_call: number of coefficients in coefficients list for call spline (rest are for put spline)
    # - all_strikes: list of all strikes (x-coordinate of points at which spline is evaluated)
    # - all_times: list of all times (y-coordinate of points at which spline is evaluated)
    # - valid_call_coordinates: 1D strike-time coordinates mask of calls that were not removed as not arbitrable
    # - valid_put_coordinates: 1D strike-time coordinates mask of puts that were not removed as not arbitrable
    # - valid_call_prices: 2D grid of original (not removed) arbitrage-free call prices
    # - valid_put_prices: 2D grid of original (not removed) arbitrage-free
    # - parity: list of put-call parity expected values (difference of put-call prices from which should be minimised)
    # - tck_call: list output from interpolate.bisplrep with knots on calls; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    # - tck_put: list output from interpolate.bisplrep with knots on puts; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev

    # Current coefficients
    new_tck_calls = tck_call[0:2] + [coefficients[:n_call]] + tck_call[3:]
    new_tck_puts = tck_put[0:2] + [coefficients[n_call:]] + tck_put[3:]

    # Newly calculated values at every grid point
    all_interp_calls = interpolate.bisplev(all_strikes, all_times, new_tck_calls)
    all_interp_puts = interpolate.bisplev(all_strikes, all_times, new_tck_puts)

    # Flatten along each column i.e. time-to-maturity first
    raveled_calls = all_interp_calls.ravel(order='F')
    raveled_puts = all_interp_puts.ravel(order='F')

    # Extract interpolated values for original, arbitrage-free points
    valid_interp_calls = raveled_calls[valid_call_coordinates]
    valid_interp_puts = raveled_puts[valid_put_coordinates]

    # Mean-squared error
    call_difference = np.sum((valid_interp_calls - valid_call_prices) ** 2)
    put_difference = np.sum((valid_interp_puts - valid_put_prices) ** 2)
    parity_difference = np.sum((raveled_calls - raveled_puts - parity) ** 2)
    return call_difference + put_difference + parity_difference


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


def index_stock_csv_input(csv_path, index_stocks=0):
    # Read the CSV file. The file has multi-level headers, hence header=list of length the number of levels.
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


def option_excel_input(excel_path):
    xl = pd.read_excel(excel_path, sheet_name=None)
    sheets = xl.keys()
    return xl
