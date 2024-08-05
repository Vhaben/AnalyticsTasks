# AnalyticsTasks

Miscellanious data analytics and prompting:
1. [Index data formatter](#index-data-formatter)
2. [Options data formatter](#options-data-formatter)

## Index data formatter

Input: index or stock or list of stocks, two dates  
Workflow:
- Downloads stock data of the period between the two dates from Yahoo! Finance API
- Cleans data
Output: csv file

## Options data formatter

Input: index or stock or list of stocks, two dates  
Workflow:
- Downloads data of the period between the two dates from Yahoo! Finance API
- Cleans data
- Checks non-arbitrage conditions and cleans data appropriately
Output: csv file
