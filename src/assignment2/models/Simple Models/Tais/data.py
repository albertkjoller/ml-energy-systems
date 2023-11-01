import pandas as pd
#%% Importing price-data
prices = pd.read_csv("/Users/tais/Downloads/ml-energy-systems/src/assignment2/data/day_ahead_prices.csv")
#%% Cleaning price-data
prices = prices[prices['PriceArea'] == 'DK1'][['StartTimeUTC','SpotPriceEUR']]
prices['StartTimeUTC'] = pd.to_datetime(prices['StartTimeUTC'])
prices = prices.set_index('StartTimeUTC')['SpotPriceEUR']