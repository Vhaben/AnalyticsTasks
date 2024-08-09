from formatter import *

def yes_no(question="Y/N"):
    while True:
        user_input = input(question)
        if user_input.lower()[0] == "y":
            answ = 1
            break
        elif user_input.lower()[0] == "n":
            answ = 0
            break
        else:
            print("Invalid input. Please enter yes/no.")
    return answ

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


def csv_input(csv):
    return pd.read_csv(csv)


def main():
    j = DataToFormat("^GSPC", "2020-01-01", "2020-01-14")
    j.csv_output(j.pandas_index_stocks)

if __name__ == "__main__":
    # a=Index("^GSPC")
    # print(a.stocks_in_index("shortName"))
    main()