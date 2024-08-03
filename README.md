# AnalyticsTasks

Miscellanious data analytics and prompting:
1. [Index data formatter](#index-data-formatter)
2. [Options data formatter](#options-data-formatter)

## Index data formatter

Input: index, two dates
Workflow:
- Finds corresponding tickers
- Downloads stocks data of interval between dates from Yahoo! Finance API
- Cleans data
Output: csv file

## Options data formatter

Input: index or stock, two dates
Workflow:
- Finds corresponding tickers, if applicable
- Downloads data of interval between dates from Yahoo! Finance API
- Cleans data
- Checks non-arbitrage conditions and cleans data appropriately
Output: csv file
