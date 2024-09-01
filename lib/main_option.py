from formatter_option import *


def main():
    apl = Option('AAPL')
    # a, b, c = apl.arbitrage_conditions()
    # apl.interp_noArb(method=1)
    # apl.excel_output(2)
    # apl.interp_noArb(method=2).to_excel('AAPL_interp_constraints.xlsx')
    # print(d)
    # apl.arbitrage_conditions().to_excel('AAPL_options.xlsx')
    apl.excel_output()
    pass


if __name__ == "__main__":
    main()
