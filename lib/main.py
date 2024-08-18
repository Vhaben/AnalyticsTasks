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
    df_2d, calls_1d, puts_1d = apl.arbitrage_conditions()
    # apl.arbitrage_conditions().to_excel('AAPL_options.xlsx')


    # -------------------------------------
    # Test interpolation by piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface (CloughTocher2DInterpolator) of grid of strikes and times to maturity for no-arbitrage call prices
    mask_calls = ~np.isnan(calls_1d['noArbitrage'].values)
    # a,b,c=calls_1d['strike'][mask_calls].tolist(), calls_1d['timeToMat'][mask_calls].tolist(), calls_1d['noArbitrage'][mask_calls].tolist()
    strikes_times_mat_tuples=list(itertools.product(apl.strikes_list, apl.times_maturity))
    markPrice_interped = griddata((calls_1d['strike'][mask_calls].tolist(),calls_1d['timeToMat'][mask_calls].tolist()), calls_1d['noArbitrage'][mask_calls].tolist(), strikes_times_mat_tuples, method='cubic')
    df=pd.DataFrame(index=apl.strikes_list, columns=apl.times_maturity)
    n_times=len(apl.times_maturity)
    for n_strike in range(len(apl.strikes_list)):
        interped_vals=markPrice_interped[n_strike*n_times:(n_strike+1)*n_times]
        df.iloc[n_strike,:]=interped_vals
    df.to_excel('AAPL_options_grid_interpolation.xlsx')
    pass


if __name__ == "__main__":
    main()
