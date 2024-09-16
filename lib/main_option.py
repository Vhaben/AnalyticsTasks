from formatter_option import *


def main():
    apl = VanillaOption('AAPL')

    # Example tests

    # Interpolation with SLSQP (least-squares) with 2021-02-10 prices, using cubic splines (recommended despite negative values)
    apl.excel_output(date_c='2021-02-10', maturity_flag=False)

    # Interpolation with trust-constr with 2021-02-10 prices
    # apl.excel_output(method=1, interpo_method='trust-constr', date_c='2021-02-10')

    # Interpolation with griddata method with 2021-02-10 prices
    # apl.excel_output(method=2, date_c='2021-02-10')

    # Deprecated : interpolation into future using YFinance prices for today and future expiration dates
    # apl.excel_output() # No date specified
    pass

if __name__ == "__main__":
    main()
