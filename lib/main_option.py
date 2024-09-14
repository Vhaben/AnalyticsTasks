from formatter_option import *


def main():
    apl = VanillaOption('AAPL')
    # a, b, c = apl.arbitrage_conditions()
    # apl.interp_noArb(method=1)
    # apl.excel_output(2)
    # apl.interp_noArb(method=2).to_excel('AAPL_interp_constraints.xlsx')
    # print(d)
    # apl.arbitrage_conditions().to_excel('AAPL_options.xlsx')
    # apl.excel_output(interpo_method='COBYLA')

    response = requests.get(
        "https://www.dolthub.com/api/v1alpha1/post-no-preference/options",
        headers={},
    )
    data = response.json()
    print(data)

    query = '''SELECT * FROM option_chain WHERE act_symbol = 'TSLA' and date = '2021-09-15' and delta > 0.45 and delta < 0.55 limit 200;'''
    res = requests.get(
        'https://www.dolthub.com/api/v1alpha1/{}/{}/{}'.format("post-no-preference", "options", "master"),
        params={'q': query},
    )
    data2=res.json()
    print(data2['rows'])
    a=pd.DataFrame(data2['rows'])
    pass


def get_option_data(stock_symbol, expiration_date, option_type, strike):
    stock = yf.Ticker(stock_symbol)
    option_chain = stock.option_chain(expiration_date)
    options = getattr(option_chain, "calls" if option_type.startswith("call") else "puts")
    option_data = options[options["strike"] == strike]
    return option_data


def get_option_history_data(contract_symbol, days_before_expiration=30):
    option = yf.Ticker(contract_symbol)
    option_info = option.info
    option_expiration_date = datetime.datetime.fromtimestamp(option_info["expireDate"])

    start_date = option_expiration_date - datetime.timedelta(days=days_before_expiration)
    option_history = option.history(start=start_date)
    return option_history

def main2(*args):
    # Example:
    stock_symbol = "AAPL"
    expiration_date = "2023-10-27"
    expiration_date = None
    option_type = "call"
    strike = 170.0

    option_data = get_option_data(stock_symbol, expiration_date, option_type, strike)
    print(option_data)
    for i, od in option_data.iterrows():
        contract_symbol = od["contractSymbol"]
        option_history = get_option_history_data(contract_symbol)
        first_option_history = option_history.iloc[0]
        first_option_history_date = option_history.index[0]
        first_option_history_close = first_option_history["Close"]
        print("For {}, the closing price was ${:.2f} on {}.".format(
            contract_symbol,
            first_option_history_close,
            first_option_history_date
        ))


if __name__ == "__main__":
    # data = end_of_day()
    # print(data.to_string())
    main()
