import pandas as pd
from pandas import date_range

from formatter_index_stock import *


# Target functions and constrain functions for optimisation using SciPy's minimize

def convexity_constraint(params, num_calls, opt_type=None):
    """
    Convexity constraint for optimization after interpolation (not together) for VanillaOption.interp_noArb method

    :param params: 1D array of calls and puts' market prices concatenated
    :param num_calls: number of calls to know last index in params array storing calls info
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put

    :return:
        None: if invalid option type (not vanilla call or put)
        1 or 0: if respects constraint
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

    return int(not (np.all(np.diff(params[:num_calls], n=2) > 0) and np.all(np.diff(params[num_calls:], n=2) > 0)))


def monoticity_constraint(params, num_calls, opt_type=None):
    """
    Monotonicity constraint for optimization after interpolation (not together) for VanillaOption.interp_noArb method

    :param params: 1D array of calls and puts' market prices concatenated
    :param num_calls: number of calls to know last index in params array storing calls info
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put

    :return:
        None: if invalid option type (not vanilla call or put)
        1 or 0: if respects constraint
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

    return int(not (np.all(np.diff(params[:num_calls], n=1) < 0)  # Call prices decreasing in strike axis
                    and np.all(np.diff(params[num_calls:], n=1) > 0)))  # Put prices increasing in strike axis


def parity_minimization(call_puts, spot_strike, opt_type=None):
    """
    Objective function for minimization of put-call parity by mean squared error for Option.interp_noArb method

    :param call_puts: list of (interpolated) calls and puts
    :param spot_strike: list of put-call parity values for all strikes and times-to-maturity
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put

    :return:
        None: if invalid option type (not vanilla call or put)
        Squared error
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

    n = len(spot_strike)
    calls = call_puts[:n]
    puts = call_puts[n:]
    return np.mean((calls - puts - spot_strike) ** 2)


def parity_min_with_interp(trial, spot_strike, interped, opt_type=None):
    """
    Objective function for minimization of put-call parity and mean squared error between interpolated values and new values for VanillaOption.interp_noArb method

    :param trial: list of (interpolated) calls and puts
    :param spot_strike: list of put-call parity values for all strikes and times-to-maturity
    :param interped: interpolated values of calls and puts
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put

    :return:
        None: if invalid option type (not vanilla call or put)
        Squared error
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

    return parity_minimization(trial, spot_strike) + np.mean((trial - interped) ** 2)


def spline_convexity(coefficients, n_call, all_strikes, all_times, valid_call_coordinates, valid_put_coordinates,
                     tck_call, tck_put, opt_type=None, calendar_spread=False):
    """
    Convexity and monotonicity constraints for iterative optimization of spline coefficients for VanillaOption.spline_interp method

    :param coefficients: coefficients of spline of call and put
    :param n_call: number of coefficients in coefficients list for call spline (rest are for put spline)
    :param all_strikes: list of all strikes (x-coordinate of points at which spline is evaluated)
    :param all_times: list of all times (y-coordinate of points at which spline is evaluated)
    :param valid_call_coordinates: 1D strike-time coordinates mask of calls that were not removed as not arbitrable
    :param valid_put_coordinates: 1D strike-time coordinates mask of puts that were not removed as not arbitrable
    :param tck_call: list output from interpolate.bisplrep with knots on calls; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    :param tck_put:list output from interpolate.bisplrep with knots on puts; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put
    :param calendar_spread: flag to choose whether to impose calendar spread constraint, i.e. first derivative in time-to-maturity

    :return:
        None: if invalid option type (not vanilla call or put)
        Minimum second derivative along strike axis
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

    # Current coefficients
    new_tck_calls = tck_call[0:2] + [coefficients[:n_call]] + tck_call[3:]
    new_tck_puts = tck_put[0:2] + [coefficients[n_call:]] + tck_put[3:]

    # Convexity as second derivative along strike axis
    d2z_dx2_calls = interpolate.bisplev(all_strikes, all_times, new_tck_calls, dx=2, dy=0)
    d2z_dx2_puts = interpolate.bisplev(all_strikes, all_times, new_tck_puts, dx=2, dy=0)

    # Flatten along each column i.e. time-to-maturity first

    # Minimum of second derivatives must be positive if convex
    min_convex = min(np.min(np.ravel(d2z_dx2_calls, order='F')[valid_call_coordinates]),
                     np.min(np.ravel(d2z_dx2_puts, order='F')[valid_put_coordinates]))

    # Monotonicity constraint as first derivative along strike axis
    dz_dx_calls = interpolate.bisplev(all_strikes, all_times, new_tck_calls, dx=1, dy=0)
    dz_dx_puts = interpolate.bisplev(all_strikes, all_times, new_tck_puts, dx=1, dy=0)

    # Maximum of first derivative in strike of calls must be negative
    # Minimum of first derivative in strike of puts must be positive
    max_call_dstrike = np.max(np.ravel(dz_dx_calls, order='F')[valid_call_coordinates])
    min_put_dstrike = np.max(np.ravel(dz_dx_puts, order='F')[valid_put_coordinates])

    # Calendar spread if chosen: first derivative along time-to-maturity axis
    if calendar_spread == True:
        dz_dy_calls = interpolate.bisplev(all_strikes, all_times, new_tck_calls, dx=0, dy=1)
        dz_dy_puts = interpolate.bisplev(all_strikes, all_times, new_tck_puts, dx=0, dy=1)

        # Minimum of first derivatives in time must be positive
        min_dtime = min(np.min(np.ravel(dz_dy_calls, order='F')[valid_call_coordinates]),
                        np.min(np.ravel(dz_dy_puts, order='F')[valid_put_coordinates]))

        return min(min_dtime, min_put_dstrike, - max_call_dstrike, min_convex)

    return min(min_put_dstrike, - max_call_dstrike, min_convex)


def bounds_vanilla_constraint(coefficients, n_call, all_strikes, all_times, call_bounds, put_bounds, tck_call, tck_put,
                              opt_type=None):
    """
    Convexity constraint for iterative optimization of spline coefficients for VanillaOption.spline_interp method

    :param coefficients: coefficients of spline of call and put
    :param n_call: number of coefficients in coefficients list for call spline (rest are for put spline)
    :param all_strikes: list of all strikes (x-coordinate of points at which spline is evaluated)
    :param all_times: list of all times (y-coordinate of points at which spline is evaluated)
    :param call_bounds: list of tuples of lower and upper bounds on call prices
    :param put_bounds: list of tuples of lower and upper bounds on put prices
    :param tck_call: list output from interpolate.bisplrep with knots on calls; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    :param tck_put: list output from interpolate.bisplrep with knots on puts; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put

    :return:
        None: if invalid option type (not vanilla call or put)
        1 or -1: 1 if respects bounds on vanilla options
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

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
                   valid_call_prices, valid_put_prices, parity, tck_call, tck_put, opt_type=None):
    """
    Objective function for iterative optimization of spline coefficients for VanillaOption.spline_interp method

    :param coefficients: coefficients of spline of call and put
    :param n_call: number of coefficients in coefficients list for call spline (rest are for put spline)
    :param all_strikes: list of all strikes (x-coordinate of points at which spline is evaluated)
    :param all_times: list of all times (y-coordinate of points at which spline is evaluated)
    :param valid_call_coordinates: 1D strike-time coordinates mask of calls that were not removed as not arbitrable
    :param valid_put_coordinates: 1D strike-time coordinates mask of puts that were not removed as not arbitrable
    :param valid_call_prices: 2D grid of original (not removed) arbitrage-free call prices
    :param valid_put_prices: 2D grid of original (not removed) arbitrage-free
    :param parity: list of put-call parity expected values (difference of put-call prices from which should be minimised)
    :param tck_call: list output from interpolate.bisplrep with knots on calls; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    :param tck_put: list output from interpolate.bisplrep with knots on puts; needed to calculate derivatives at the non-arbitrable points using interpolate.bisplev
    :param opt_type: flag to ensure function only runs if type is a vanilla call or put

    :return:
        None: if invalid option type (not vanilla call or put)
        Total squared error
    """

    if opt_type not in ["vanilla", "vanilla_call", "vanilla_put"]:
        return None

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


class Option(YfInput):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.type_opt = None  # Option type


class VanillaOption(Option):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.type_opt = "vanilla"

        self.expiration_dates = self.yf_ticker.options  # Future expiration dates

        try:
            self.currency = self.yf_ticker.info['currency']
            if self.currency in rf_currencies:
                self.rfr = rf_currencies[self.currency] / 100
            if self.currency in timezone_offset:
                self.tzone = timezone_offset[self.currency]
        except:
            self.currency = None
            self.rfr = None
            self.tzone = None

        self.times_maturity = []
        self.strikes_list = []

        try:
            self.underlying = self.yf_ticker.history(
                start=(datetime.datetime.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')).Close.iloc[0]
        except:
            self.underlying = None

        self.epsilon_parity = 0.05  # Put-call parity within epsilon fraction of spot price
        self.delta = self.epsilon_parity * self.underlying  # Put-call parity bounds for equality (plus or minus epsilon currency)

        self.data_source = ""  # Source of price data: YFinance or DoltHub

    def options_data_dolt(self, start_d: str, end_d: str) -> DataFrame:
        """
        Source option price data from DoltHub database using SQL query

        :param start_d: start date of window of option price data
        :param end_d: end date of window of option price data
        :return: 2D DataFrame with strikes as index and times-to-maturity as MultiIndex column
        """

        if self.data_source == "YF":
            print("Data already sourced from YFinance. Overwritting YFinance data with DoltHub data.")
        self.data_source = "DoltHub"
        self.times_maturity = []

        # Format dates to Y-M-D
        start_d = date_formatter(start_d)
        end_d = date_formatter(end_d)
        if start_d is None or end_d is None:
            print("Invalid dates.")

        # Write SQL query from DoltHub
        query_date = ""
        for date_i in pd.date_range(start=start_d, end=end_d):
            query_date += "date = '" + date_i.to_pydatetime().strftime('%Y-%m-%d') + "' or "
        query_date = query_date[:-4]
        query = "SELECT * FROM option_chain WHERE (act_symbol = '" + self.symbol + "') and (" + query_date + ")"

        res = requests.get(
            'https://www.dolthub.com/api/v1alpha1/{}/{}/{}'.format("post-no-preference", "options", "master"),
            params={'q': query},
        )
        data = res.json()
        df = pd.DataFrame(data['rows'])

        # Format DataFrame
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['date'] = pd.to_datetime(df['date'])
        df['bid'] = pd.to_numeric(df['bid'])
        df['ask'] = pd.to_numeric(df['ask'])
        df['strike'] = pd.to_numeric(df['strike'])

        # Drop expiration dates in the future
        df.drop(df[df.expiration > pd.Timestamp('today').floor('D')].index, inplace=True)

        # Drop unnecessary data
        df.drop(columns=['delta','gamma','theta','vega','rho'],inplace=True)

        # Market price as mean of bid and ask
        df['markPrice'] = (df['bid'] + df['ask']) / 2

        df['time_to_mat'] = (df['expiration'] - df['date'])/ pd.Timedelta('365 days')

        self.times_maturity = sorted(df['time_to_mat'].unique())
        self.strikes_list = sorted(df['strike'].unique())

        # 2D DataFrame data
        df_list = []

        for time_to in self.times_maturity:
            df_time_to_call= df[(df.time_to_mat==time_to) & (df.call_put=='Call')]
            df_time_to_put = df[(df.time_to_mat==time_to) & (df.call_put=='Put')]
            # df_time_to_call.columns = pd.MultiIndex.from_tuples([('Call',) + _ for _ in df_time_to_call.columns])
            # df_time_to_put.columns = pd.MultiIndex.from_tuples([('Put',) + _ for _ in df_time_to_put.columns])

            # Set index as strike to create 2D DataFrame
            df_time_to_call.set_index('strike', inplace=True)
            df_time_to_put.set_index('strike', inplace=True)

            # Merge calls and puts into a single dataframe with MultiIndex: level 0 for 'Call' and 'Put'; level 1 for subitems, e.g. 'markPrice'
            df_time_to = pd.concat(objs=[df_time_to_call, df_time_to_put], axis=1, keys=['Call', 'Put'], )

            # Add level 0 of MultiIndex: time to maturity
            df_time_to.columns = pd.MultiIndex.from_tuples([(time_to,) + _ for _ in df_time_to.columns])
            # Format 2D DataFrame


            df_list.append(df_time_to)

        '''Testing: multiple data points with same strike and time-to-maturity as different dates'''
        for dftest in df_list:
            print(dftest.index.is_unique)

        # Define 2D DataFrame for calls and puts: strikes increase in rows; times to expiration increase along columns
        options_df_2d = pd.concat(df_list, axis=1)
        options_df_2d.sort_index(inplace=True)
        self.strikes_list = options_df_2d.index.values
        return options_df_2d

    def options_expiration(self):
        """
        Source option price data from YFinance database for future expiration dates only

        :return: 2D DataFrame with strikes as index and times-to-maturity as MultiIndex column
        """
        if len(self.expiration_dates) == 0:
            return None
        else:
            if self.data_source == "DoltHub":
                print("Data already sourced from DoltHub. Overwritting DoltHub data with YFinance data.")
            self.data_source = "YF"
            self.times_maturity = []

            # 2D DataFrame data
            df_list = []

            # 1D DataFrames data
            # calls_df_data = {'strike': [], 'timeToMat': [], 'markPrice': []}
            # puts_df_data = {'strike': [], 'timeToMat': [], 'markPrice': []}
            self.times_maturity = []
            for exp_date in self.expiration_dates:
                # Get option data from YFinance API (opt_chain is a list of dataframes)
                opt_chain = self.yf_ticker.option_chain(exp_date)

                # Get DataFrame for calls and puts
                calls = opt_chain.calls
                puts = opt_chain.puts

                # Calculate market price as average
                calls['markPrice'] = (calls['bid'] + calls['ask']) / 2
                puts['markPrice'] = (puts['bid'] + puts['ask']) / 2

                # Calculate time to maturity in years: to second accuracy until midnight of expiration date (add 1 day to expiration to set to end of day)
                time_to_mat = (pytz.timezone(self.tzone).localize(
                    datetime.datetime.strptime(exp_date, "%Y-%m-%d")
                    + datetime.timedelta(hours=24)
                )
                               - datetime.datetime.now(pytz.timezone(self.tzone))).total_seconds() / (
                                      24 * 60 * 60) / 360

                self.times_maturity.append(time_to_mat)

                ''' No longer in use
                # Append market price and corresponding strike to dictionary for 1D DataFrames
                calls_df_data['markPrice'] += calls['markPrice'].tolist()
                calls_df_data['strike'] += calls['strike'].tolist()
                calls_df_data['timeToMat'] += [time_to_mat] * len(calls['strike'])
                puts_df_data['markPrice'] += puts['markPrice'].tolist()
                puts_df_data['strike'] += puts['strike'].tolist()
                puts_df_data['timeToMat'] += [time_to_mat] * len(puts['strike'])
                '''

                # Set index as strike to create 2D DataFrame
                calls.set_index('strike', inplace=True)
                puts.set_index('strike', inplace=True)

                # Format 2D DataFrame
                # Merge calls and puts into a single dataframe with MultiIndex: level 0 for 'Call' and 'Put'; level 1 for subitems, e.g. 'markPrice'
                options = pd.concat(objs=[calls, puts], axis=1, keys=['Call', 'Put'], )
                options.drop(
                    columns=list(itertools.product(['Call', 'Put'],
                                                   ['contractSize', 'currency', 'change', 'percentChange',
                                                    'lastTradeDate',
                                                    'lastPrice', 'openInterest', 'inTheMoney'])),
                    inplace=True)
                options['expirationDate'] = exp_date
                options['expirationDate'] = pd.to_datetime(options['expirationDate'])

                # Add level 0 of MultiIndex: time to maturity
                options.columns = pd.MultiIndex.from_tuples([(time_to_mat,) + _ for _ in options.columns])
                df_list.append(options)

            ''' No longer in use
            # Define 1D DataFrames for calls and puts
            calls_df_1d = pd.DataFrame(calls_df_data)
            puts_df_1d = pd.DataFrame(puts_df_data)
            '''

            # Define 2D DataFrame for calls and puts: strikes increase in rows; times to expiration increase along columns
            options_df_2d = pd.concat(df_list, axis=1)
            options_df_2d.sort_index(inplace=True)
            self.strikes_list = options_df_2d.index.values

            ''' No longer in use
            # Define separate DataFrames with only call and put values
            calls_2d = options_df_2d[[(_, 'Call', 'markPrice') for _ in self.times_maturity]].copy()
            puts_2d = options_df_2d[[(_, 'Put', 'markPrice') for _ in self.times_maturity]].copy()
            calls_2d.columns = self.times_maturity
            puts_2d.columns = self.times_maturity
            '''
            return options_df_2d

    def arbitrage_conditions(self, start_date=None,end_date=None):
        """
        Cleans sourced data in DataFrames by removing arbitrable values (checks no-arbitrage conditions); passed for interpolation to interp_noArb method

        Input:
            DataFrames: automatically inputted from options_expiration method or options_data_dolt method depending on choice

        :param start_date: optional argument for start of price window if using DoltHub database for option prices; default is None i.e. using YFinance
        :param end_date: optional argument for end of price window if using DoltHub database for option prices; default is None i.e. using YFinance

        :return:
            DataFrame with all info (calls and puts' original market prices, strikes, etc.)
            DataFrame with calls' cleaned values in 1D
            DataFrame with puts' cleaned values in 1D
        """
        if self.data_source == "DoltHub" and start_date is not None and end_date is not None:
            df = self.options_data_dolt(start_d=start_date, end_d=end_date)
        else:
            df = self.options_expiration()
        df = df.dropna(how='all', axis=1)
        df = df.dropna(how='all', axis=0)

        strike_index = df.index
        strike_delta_series = strike_index.to_series().diff()
        n_strikes = len(strike_index)

        # Lists of market prices with arbitrable values removed for 1D DataFrames
        # calls_markPrice_rem = []
        # puts_markPrice_rem = []
        calls_df_data = {'strike': [], 'timeToMat': [], 'noArbitrage': [], 'unclean': []}
        puts_df_data = {'strike': [], 'timeToMat': [], 'noArbitrage': [], 'unclean': []}

        for time_to in self.times_maturity:
            # Define new DataFrame to avoid warnings about appending too many columns to existing DataFrame df
            extra_data = pd.DataFrame(index=strike_index)
            extra_data[time_to, 'parity', ''] = self.underlying - strike_index * np.exp(-self.rfr * time_to)
            # extra_data[time_to, 'delta', ''] = df[time_to, 'Call', 'markPrice'] - df[time_to, 'Put', 'markPrice']

            # First partial derivatives
            extra_data[time_to, 'Call', 'D1'] = df[time_to, 'Call', 'markPrice'].diff() / strike_delta_series
            extra_data[time_to, 'Put', 'D1'] = df[time_to, 'Put', 'markPrice'].diff() / strike_delta_series
            # Second partial derivatives
            extra_data[time_to, 'Call', 'D2'] = extra_data[time_to, 'Call', 'D1'].diff() / strike_delta_series
            extra_data[time_to, 'Put', 'D2'] = extra_data[time_to, 'Put', 'D1'].diff() / strike_delta_series

            # Append calculated data back to original DataFrame
            df = pd.concat([df, extra_data], axis=1)

            # Keep unclean data for reference
            df[time_to, 'Call', 'unclean'] = df[time_to, 'Call', 'markPrice']
            df[time_to, 'Put', 'unclean'] = df[time_to, 'Put', 'markPrice']

            # Put-Call parity
            df.loc[(df[time_to, 'Call', 'markPrice'] - df[time_to, 'Put', 'markPrice'] - df[
                time_to, 'parity', '']).abs() > self.delta, (time_to, 'Call', 'markPrice')] = np.nan

            # Bounds on vanilla options
            df.loc[(np.maximum(0, df[time_to, 'parity', '']) > df[time_to, 'Call', 'markPrice']) | (
                    df[time_to, 'Call', 'markPrice'] > self.underlying), (time_to, 'Call', 'markPrice')] = np.nan
            df.loc[(np.maximum(0, -df[time_to, 'parity', '']) > df[time_to, 'Put', 'markPrice']) |
                   (df[time_to, 'Put', 'markPrice'] > - df[time_to, 'parity', ''] + self.underlying), (
                time_to, 'Put', 'markPrice')] = np.nan

            # Convexity
            df.loc[df[time_to, 'Call', 'D2'] < 0, (time_to, 'Call', 'markPrice')] = np.nan
            df.loc[df[time_to, 'Put', 'D2'] < 0, (time_to, 'Put', 'markPrice')] = np.nan
            # N.B. convexity data cleaning does not remove the entire column of market prices for a given time-to-maturity (which would be more rigorous),
            # only the values where it isn't convex

            ''' !!! No longer in use
            # Remove arbitrable values from 1D DataFrames for calls and puts
            # Find indices that weren't removed by checking have a contract symbol
            calls_markPrice_rem += df.loc[
                df[time_to, 'Call', 'contractSymbol'].notna(), (time_to, 'Call', 'markPrice')].tolist()
            puts_markPrice_rem += df.loc[
                df[time_to, 'Put', 'contractSymbol'].notna(), (time_to, 'Put', 'markPrice')].tolist()
            '''

            # Append market price and corresponding strike to dictionary for 1D DataFrames
            calls_df_data['noArbitrage'] += df[(time_to, 'Call', 'markPrice')].tolist()
            calls_df_data['strike'] += strike_index.tolist()
            calls_df_data['timeToMat'] += [time_to] * n_strikes
            calls_df_data['unclean'] += df[(time_to, 'Call', 'unclean')].tolist()
            puts_df_data['noArbitrage'] += df[(time_to, 'Put', 'markPrice')].tolist()
            puts_df_data['strike'] += strike_index.tolist()
            puts_df_data['timeToMat'] += [time_to] * n_strikes
            puts_df_data['unclean'] += df[(time_to, 'Put', 'unclean')].tolist()

        # Update 2D DataFrame again and strikes_list if any strike rows are completely empty
        # df = df.dropna(subset=[(_,opt_type,'markPrice') for _ in self.times_maturity for opt_type in ['Call','Put']], how='all', axis=0)
        self.strikes_list = df.index.values

        ''' !!! No longer in use
        # Redefine 1D DataFrames for calls and puts with only arbitrage-free values
        calls_df_1d['noArbitrage'] = calls_markPrice_rem
        puts_df_1d['noArbitrage'] = puts_markPrice_rem
        '''

        # Define 1D DataFrames for calls and puts
        calls_df_1d = pd.DataFrame(calls_df_data)
        puts_df_1d = pd.DataFrame(puts_df_data)

        return df, calls_df_1d, puts_df_1d

    def interp_noArb(self, started=None,ended=None, method=1):
        """
        Interpolates option surface

        Automatic input: output of arbitrage_conditions method, i.e. 3 cleaned DataFrames, i.e. DataFrames with arbitrable values replaced by NaN

        :param method (int): choice of interpolation method; default of 1 interpolates separately for each time-to-maturity; method 2 interpolates 2D grid at once
        :return:
            DataFrame with all info (calls and puts' original market prices, strikes, etc.)
            DataFrame with calls' interpolated values of option surface such that respect no-arbitrage conditions
            DataFrame with puts' interpolated values of option surface such that respect no-arbitrage conditions
        """
        if self.data_source == "DoltHub" and started is not None and ended is not None:
            df, calls_df_1d, puts_df_1d = self.arbitrage_conditions(start_date=started,end_date=ended)
        else:
            df, calls_df_1d, puts_df_1d = self.arbitrage_conditions()

        # df = df.dropna(how='all', axis=1)
        # df = df.dropna(how='all', axis=0)
        # Extra columns: ['contractSymbol','bid','ask','volume','impliedVolatility'] ['expirationDate']

        strike_index = df.index
        n_calls = len(strike_index)
        strike_delta_series = strike_index.to_series().diff()
        if method == 1:
            # Interpolation by Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) + constraints by minimisation: 1D monotonic
            for time_to in self.times_maturity:
                interp = 0

                mask_calls = ~np.isnan(df[time_to, 'Call', 'markPrice'].values)
                if sum(mask_calls) > 1:
                    interp += 1

                    monotonic_interpol = interpolate.PchipInterpolator(self.strikes_list[mask_calls],
                                                                       df[time_to, 'Call', 'markPrice'].values[
                                                                           mask_calls])
                    calls = monotonic_interpol(self.strikes_list)
                    df[time_to, 'Call', 'markPrice'] = calls

                mask_puts = ~np.isnan(df[time_to, 'Put', 'markPrice'].values)
                if sum(mask_puts) > 1:
                    interp += 1

                    monotonic_interpol = interpolate.PchipInterpolator(self.strikes_list[mask_puts],
                                                                       df[time_to, 'Put', 'markPrice'].values[
                                                                           mask_puts])
                    puts = monotonic_interpol(self.strikes_list)
                    df[time_to, 'Put', 'markPrice'] = puts


        elif method == 2:
            # Interpolation by piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface (CloughTocher2DInterpolator)
            # of grid of strikes and times to maturity for no-arbitrage call prices

            # Calls
            mask_calls = ~np.isnan(calls_df_1d['noArbitrage'].values)

            # Coordinates of points at which to evaluate
            strikes_times_mat_tuples = list(itertools.product(self.strikes_list, self.times_maturity))

            calls_markPrice_interped = interpolate.griddata(
                (calls_df_1d['strike'][mask_calls].tolist(), calls_df_1d['timeToMat'][mask_calls].tolist()),
                calls_df_1d['noArbitrage'][mask_calls].tolist(), strikes_times_mat_tuples, method='cubic')

            # Puts
            mask_puts = ~np.isnan(puts_df_1d['noArbitrage'].values)

            puts_markPrice_interped = interpolate.griddata(
                (puts_df_1d['strike'][mask_puts].tolist(), puts_df_1d['timeToMat'][mask_puts].tolist()),
                puts_df_1d['noArbitrage'][mask_puts].tolist(), strikes_times_mat_tuples, method='cubic')

            n_times = len(self.strikes_list)
            for i, time_to in enumerate(self.times_maturity):
                df[time_to, 'Call', 'markPrice'] = calls_markPrice_interped[i * n_times:(i + 1) * n_times]
                df[time_to, 'Put', 'markPrice'] = puts_markPrice_interped[i * n_times:(i + 1) * n_times]
            interp = 2

        if interp == 2:  # If there are values to be interpolated
            # Impose non-arbitrage constraints by minimisation

            for time_to in self.times_maturity:
                pc_parity = (self.underlying - strike_index * np.exp(-self.rfr * time_to))

                # Bounds on vanilla options
                lower_call = np.maximum(0, pc_parity)
                bounds_call = tuple(itertools.product(lower_call, [float(self.underlying)]))

                lower_put = np.maximum(0, -pc_parity)
                upper_put = self.underlying - pc_parity
                bounds_put = tuple(zip(lower_put, upper_put))

                bounds = bounds_call + bounds_put

                # Retrieve interpolated values
                calls = df[time_to, 'Call', 'markPrice']
                puts = df[time_to, 'Put', 'markPrice']
                calls_puts = np.concatenate((calls, puts))

                # Convexity and monoticity constraints
                deriv_constraint = [{'type': 'eq', 'fun': convexity_constraint, 'args': (n_calls, self.type_opt)},
                                    {'type': 'eq', 'fun': monoticity_constraint, 'args': (n_calls, self.type_opt)}]

                # Minimize delta from put-call parity while enforcing convexity and bounds
                result = minimize(parity_min_with_interp, calls_puts, bounds=bounds,
                                  args=(pc_parity.values, calls_puts, self.type_opt),
                                  constraints=deriv_constraint
                                  # method="cobyla"
                                  )
                df[time_to, 'Call', 'markPrice'] = result.x[:len(calls)]
                df[time_to, 'Put', 'markPrice'] = result.x[len(calls):]

            # Extract only columns of interpolated market price for calls and puts separately
            calls_2d = df[[(_, 'Call', 'markPrice') for _ in self.times_maturity]].copy()
            puts_2d = df[[(_, 'Put', 'markPrice') for _ in self.times_maturity]].copy()
            calls_2d.columns = self.times_maturity
            puts_2d.columns = self.times_maturity
            calls_2d.index = self.strikes_list
            puts_2d.index = self.strikes_list
            return df, calls_2d, puts_2d

        else:
            return None

    def spline_interp(self, interp_method='SLSQP', calendar_spr=False):
        """
        Interpolate option surface by optimization: use bicubic spline as initial guess, then adjust coefficients to enforce constraints

        Input: 3 cleaned DataFrames, i.e. DataFrames with arbitrable values replaced by NaN, i.e. output of arbitrage_conditions method

        :param interp_method: choice of SciPy's optimisation algorithm; valid choices (as require constraints): 'COBYLA', 'COBYQA', 'SLSQP' (default), 'trust-constr'
        :param calendar_spr: optional flag if also want to check calendar spread (derivative in times-to-maturity); default is not to
        :return: returns interpolated values of option surface such that respect no-arbitrage conditions
        """

        # Check interpolation method is valid
        chosen_interp_method = valid_input(['COBYLA', 'COBYQA', 'SLSQP', 'trust-constr'], interp_method)

        # Get cleaned DataFrames of option market prices
        df, calls_df_1d, puts_df_1d = self.arbitrage_conditions()

        # Axis info
        strike_index = df.index
        n_strikes = len(self.strikes_list)
        n_times = len(self.times_maturity)

        # Interpolation by rectangular bicubic spline on grid of strikes and times to maturity for no-arbitrage call and put prices

        # Coordinates of points at which to evaluate
        # Increases in strikes first (increasing rows), then times (columns) once reached max strike
        strikes_times_mat_tuples = np.array(list(itertools.product(self.strikes_list, self.times_maturity)))

        # Calls
        mask_calls = ~np.isnan(
            calls_df_1d['noArbitrage'].values)  # Increases in strikes first (increasing rows), then times (columns)
        valid_call_strikes = calls_df_1d['strike'][mask_calls]
        valid_call_times = calls_df_1d['timeToMat'][mask_calls]
        valid_call_strike_times = np.array(list(
            itertools.product(valid_call_strikes, valid_call_times)))  # list of tuples of arbitrage-free coordinates
        # valid_call_coordinates = np.transpose(
        #     np.reshape(mask_calls, (-1, n_strikes)))  # mask of valid coordinates for grid
        valid_call_prices = calls_df_1d['noArbitrage'][mask_calls]
        # call_prices_grid = np.transpose(np.reshape(calls_df_1d['noArbitrage'], (-1, n_strikes)))[valid_call_coordinates]

        # Knots and coefficients of bivariate cubic interpolation
        # interpolate.bisplrep takes as input x[i], y[i], z[i] for each i-th point
        tck_calls = interpolate.bisplrep(valid_call_strikes,
                                         valid_call_times,
                                         valid_call_prices, kx=3, ky=3)
        call_coeffs = tck_calls[2]
        n_call_coeffs = len(call_coeffs)

        # interpolate.bisplev takes as input x-coordinates of entire grid and y-positions of entire grid (not xy-coordinates of each point)
        interped_calls = interpolate.bisplev(self.strikes_list, self.times_maturity, tck_calls)

        # Puts
        mask_puts = ~np.isnan(
            puts_df_1d['noArbitrage'].values)  # Increases in strikes first (increasing rows), then times (columns)
        valid_put_strikes = puts_df_1d['strike'][mask_puts].tolist()
        valid_put_times = puts_df_1d['timeToMat'][mask_puts].tolist()
        valid_put_strike_times = np.array(
            list(itertools.product(valid_put_strikes, valid_put_times)))  # list of tuples of arbitrage-free coordinates
        # valid_put_coordinates = np.transpose(
        #     np.reshape(mask_calls, (-1, n_strikes)))  # mask of valid coordinates for grid
        valid_put_prices = puts_df_1d['noArbitrage'][mask_puts].tolist()
        # put_prices_grid = np.transpose(np.reshape(calls_df_1d['noArbitrage'], (-1, n_strikes)))[valid_put_coordinates]

        tck_puts = interpolate.bisplrep(valid_put_strikes,
                                        valid_put_times,
                                        valid_put_prices, kx=3, ky=3)
        put_coeffs = tck_puts[2]
        n_put_coeffs = len(put_coeffs)

        interped_puts = interpolate.bisplev(self.strikes_list, self.times_maturity, tck_puts)

        # Initial guess for spline coefficients
        all_coefs = np.concatenate((call_coeffs, put_coeffs))

        # Put-call parity
        pc_parity = list(map(lambda x: self.underlying - x[0] * np.exp(-self.rfr * x[1]), strikes_times_mat_tuples))
        # Increases in strikes first, then times (once reached max strike)

        # Bounds on vanilla options
        lower_call = np.maximum(0, pc_parity)
        bounds_call = tuple(itertools.product(lower_call, [float(self.underlying)]))

        lower_put = np.maximum(0, -np.array(pc_parity))
        upper_put = self.underlying - pc_parity
        bounds_put = tuple(zip(lower_put, upper_put))

        # First choices for optimisation algorithm: COBYLA or SLPSQ; both can take a dict of constraints
        if chosen_interp_method in ['COBYLA', 'SLSQP']:

            # Constraints definitions
            constraints = [
                {'type': 'ineq', 'fun': spline_convexity,
                 'args': (
                     n_call_coeffs, self.strikes_list, self.times_maturity, mask_calls, mask_puts,
                     tck_calls, tck_puts, self.type_opt, calendar_spr)},
                {'type': 'ineq', 'fun': bounds_vanilla_constraint,
                 'args': (
                     n_call_coeffs, self.strikes_list, self.times_maturity, bounds_call, bounds_put, tck_calls,
                     tck_puts, self.type_opt)}
            ]

        # Second choices for optimisation algorithm: COBYQA or trust-constr; both can take a list of constraints SciPy constraint objects
        elif chosen_interp_method in ['COBYQA', 'trust-constr']:

            # Constraints definitions (list of SciPy constraint objects)
            constraints = [NonlinearConstraint(
                lambda x: spline_convexity(x, n_call_coeffs, self.strikes_list, self.times_maturity, mask_calls,
                                           mask_puts,
                                           tck_calls, tck_puts, self.type_opt, calendar_spr), 0, np.inf),
                NonlinearConstraint(lambda x: bounds_vanilla_constraint(x, n_call_coeffs, self.strikes_list,
                                                                        self.times_maturity, bounds_call,
                                                                        bounds_put, tck_calls,
                                                                        tck_puts, self.type_opt), 0, np.inf)]

        # Perform the optimization
        result = minimize(optimal_spline, all_coefs, args=(
            n_call_coeffs, self.strikes_list, self.times_maturity, mask_calls, mask_puts,
            valid_call_prices, valid_put_prices, pc_parity, tck_calls, tck_puts, self.type_opt),
                          constraints=constraints, method=chosen_interp_method)

        # Extract the optimized coefficients
        optimized_coefficients = result.x
        optimized_tck_call = tck_calls[0:2] + [optimized_coefficients[:n_call_coeffs]] + tck_calls[3:]
        optimized_tck_put = tck_puts[0:2] + [optimized_coefficients[n_call_coeffs:]] + tck_puts[3:]

        # Interpolate with these coefficients
        interped_calls = interpolate.bisplev(self.strikes_list, self.times_maturity, optimized_tck_call)
        interped_puts = interpolate.bisplev(self.strikes_list, self.times_maturity, optimized_tck_put)

        # Store 2D DataFrames for calls and puts
        calls_2d = pd.DataFrame(interped_calls, columns=self.times_maturity, index=self.strikes_list)
        puts_2d = pd.DataFrame(interped_puts, columns=self.times_maturity, index=self.strikes_list)

        return df, calls_2d, puts_2d

    def excel_output(self, method=0, interpo_method='SLSQP',maturity_flag=False):
        if method == 1 or method == 2:
            df, calls_2d, puts_2d = self.interp_noArb(method=method)
            exc_name= self.symbol + '_call_puts_' + interpo_method + str(method) + '.xlsx'
        else:
            df, calls_2d, puts_2d = self.spline_interp(interp_method=interpo_method,calendar_spr=maturity_flag)
            exc_name = self.symbol + '_call_puts_' + interpo_method + '_spline_cal_spr' + str(maturity_flag) + '.xlsx'
        with pd.ExcelWriter(exc_name) as writer:
            calls_2d.to_excel(writer, sheet_name='Calls')
            puts_2d.to_excel(writer, sheet_name='Puts')

