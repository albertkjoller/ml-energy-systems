import os
import json
import zipfile
from pathlib import Path
from typing import List

import pytz
from dateutil.parser import parse

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

class DataProcessor:

    def __init__(self, DATA_DIR: Path, SAVE_DIR: Path):
        self.DATA_DIR = DATA_DIR
        self.SAVE_DIR = SAVE_DIR

    # TODO: find out whether the datetime object of the power production should be associated to StartTimeUTC or EndTimeUTC: 
    ### both for the actual power and for the day ahead prices
    def load_actual_wind_power(self):
        print("Loading actual wind power production...")

        # Read csv-file
        actual_wind_power                   = pd.read_csv(self.DATA_DIR / 'raw/Actual wind power.csv', sep=';')
        # Parse datetime by combining date and hour information
        actual_wind_power['StartTimeUTC']   = pd.to_datetime(actual_wind_power['Date'] + ' ' + actual_wind_power['Time'], format='mixed')
        # Add timezone to datetime element
        actual_wind_power['StartTimeUTC']   = actual_wind_power.StartTimeUTC.dt.tz_localize(pytz.UTC) 

        # Assume that the timestamp is from DK (since it starts at hour 0 of 2021 which the data from the other files does in DK time.
        # For this reason we adjust the timeseries and express everything in terms of UTC timestamps 
        actual_wind_power['StartTimeUTC']   = actual_wind_power.StartTimeUTC - pd.to_timedelta('2 hours')

        # Get rid of redundant columns
        actual_wind_power = actual_wind_power.drop(columns=['Date', 'Time'])
        return actual_wind_power

    def load_balancing_prices(self):
        print("Loading balancing prices...")

        # Load information about up- and down-regulation prices for both 2021 and 2022
        for year_idx, year in enumerate([2021, 2022]):
            for i, filename in enumerate([f'Down-regulation price_{year}.csv', f'Up-regulation price_{year}.csv']):
                # Determine filetype
                price_type          = filename.split('-')[0]
                price_column_name   = 'Up-regulating' if price_type == 'Up' else 'Down-regulation' 
                price_column_name   = f'"{price_column_name} price in the Balancing energy market"""'
                
                # Read csv-file as temporary dataframe
                df_price_ = pd.read_csv(self.DATA_DIR / f'raw/{filename}', sep=',"', engine='python')
            
                # Handle encoding with quotation marks
                df_price_['StartTimeUTC']   = pd.to_datetime(df_price_['"Start time UTC'].str.strip('"')).dt.tz_localize(pytz.UTC)
                df_price_['EndTimeUTC']     = pd.to_datetime(df_price_['"End time UTC""'].str.strip('"')).dt.tz_localize(pytz.UTC)

                # Change datatype of prices from string to float
                df_price_[f'BalancingMarketPrice_{price_type}Reg'] = df_price_[price_column_name].str.strip('"')
                df_price_[f'BalancingMarketPrice_{price_type}Reg'] = df_price_[f'BalancingMarketPrice_{price_type}Reg'].astype(float)
            
                # Restrict data to relevant information - Danish timezone is implicitly contained in UTC timestamp
                df_price_ = df_price_[['StartTimeUTC', 'EndTimeUTC', f'BalancingMarketPrice_{price_type}Reg']]


                # Combine dataframes for both years 
                prices_ = df_price_ if i == 0 else prices_.merge(df_price_, on=['StartTimeUTC', 'EndTimeUTC'], how='outer')
                
            # Merge prices from year with currently stored price information into combined dataframe
            balancing_prices = prices_ if year_idx == 0 else pd.concat([balancing_prices, prices_], axis=0).reset_index(drop=True)
            
        return balancing_prices

    # TODO: find out whether the datetime object of the power production should be associated to StartTimeUTC or EndTimeUTC: 
    ### both for the actual power and for the day ahead prices
    def load_day_ahead_prices(self):
        print("Loading day-ahead prices...")

        # Read day ahead prices from excel sheet
        day_ahead_prices = pd.read_excel(self.DATA_DIR / 'raw/Day-ahead price.xlsx')

        # Represent time as datetime object
        day_ahead_prices['StartTimeUTC'] = pd.to_datetime(day_ahead_prices['HourUTC']).dt.tz_localize(pytz.UTC)

        # Get rid of redundant information
        day_ahead_prices = day_ahead_prices[['StartTimeUTC', 'PriceArea', 'SpotPriceDKK', 'SpotPriceEUR']]
        return day_ahead_prices

    def preprocess_climate_data(self, year):
        ### Preprocess the raw climate data and saves restructured csv file that can then be restricted to e.g. Roskilde municipality
        # Open zip folder
        zip = zipfile.ZipFile(self.DATA_DIR / f'raw/Climate data_{year}.zip')

        # Extract information from all files within the zip-folder
        weather_information = []
        for i, filename in enumerate(tqdm(zip.namelist(), desc=f'Processing all weather data from {year}')):
            # Open file
            f = zip.open(filename, 'r')
            for line in f:
                # Read each entry individually
                information = json.loads(line)
                
                # Restrict the extracted information to temporal, location and weather-related information only 
                new_observation = [
                    information['properties']['from'], 
                    information['properties']['to'], 
                    information['properties']['parameterId'], 
                    information['properties']['value'], 
                    information['geometry']['coordinates'][0], 
                    information['geometry']['coordinates'][1], 
                    information['properties']['municipalityName']
                ]
                weather_information.append(new_observation)

        # Create dataframe of weather from the given year
        column_names        = ['StartTimeUTC', 'EndTimeUTC', 'WeatherAttribute', 'Value', 'Longitude', 'Latitude', 'Municipality'] 
        weather_information = pd.DataFrame(weather_information, columns=column_names)
        
        # Save information
        weather_information.to_csv(self.SAVE_DIR / f'weather_information_{year}.csv')  

    def load_weather_from_municipality(self, municipality: str):
        print(f"Loading weather data from {municipality}...")
        # If preprocessing has not been run, do it!
        for year in [2021, 2022]:
            if not os.path.isfile(self.SAVE_DIR / f'weather_information_{year}.csv'):
                self.preprocess_climate_data(year) # preprocessing function

        # Load the saved weather information and average it
        weather = pd.DataFrame()
        for year in [2021, 2022]:
            print(f"\t --> Loading {year}...")
            # Load "preprocessed" DMI data
            weather_ = pd.read_csv(self.SAVE_DIR / f'weather_information_{year}.csv', index_col=0)
            
            # Restrict weather information to wind-related attributes
            weather_ = weather_.query(
                'WeatherAttribute == "mean_wind_speed" or WeatherAttribute == "mean_wind_dir" or WeatherAttribute == "max_wind_speed_10min" or WeatherAttribute == "max_wind_speed_3sec"'
            )

            # Restrict weather measures to location of Roskilde (as this is where the actual power production) is from
            weather_ = weather_.query(f'Municipality == "{municipality}"')

            # Merge weather information from both years
            weather = pd.concat([weather, weather_], axis=0).reset_index(drop=True)

        # Map timestrings as timestamps
        weather['StartTimeUTC'] = pd.to_datetime(weather['StartTimeUTC'].progress_apply(lambda x: parse(x)), utc=True)
        weather['EndTimeUTC']   = pd.to_datetime(weather['EndTimeUTC'].progress_apply(lambda x: parse(x)), utc=True)

        # Remove hours where there for some reason are 
        weather     = weather[weather['StartTimeUTC'].dt.microsecond == 0].sort_values('StartTimeUTC').reset_index(drop=True)
        avg_weather = weather.groupby(by=['StartTimeUTC', 'EndTimeUTC', 'WeatherAttribute'])['Value'].mean().reset_index()
        avg_weather = avg_weather.pivot(index=['StartTimeUTC', 'EndTimeUTC'], columns=['WeatherAttribute'], values='Value').reset_index()
        return avg_weather
    
    def combine_and_save_data_sources(self, weather_municipality: str = 'Roskilde', priceareas: List[str] = ['DK2']):
        
        # Load and do initial processing of data files
        actual_wind_power   = self.load_actual_wind_power()
        balancing_prices    = self.load_balancing_prices()
        day_ahead_prices    = self.load_day_ahead_prices()
        avg_weather         = self.load_weather_from_municipality(weather_municipality)

        # Merge data sources based on temporal information and pricearea
        dataset = actual_wind_power.merge(day_ahead_prices, on='StartTimeUTC', how='left')
        dataset = dataset.merge(balancing_prices, on='StartTimeUTC')

        # TODO: consider what to do with summer/wintertime hours - here we take the mean
        dataset = dataset.groupby(by=['StartTimeUTC', 'EndTimeUTC', 'PriceArea']).mean().reset_index()
        
        # Add weather information to merged dataset
        dataset = dataset.merge(avg_weather, on=['StartTimeUTC', 'EndTimeUTC'])

        ### SAVE DATAFRAMES ###
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        # Save loaded and combined data files
        for pricearea in priceareas:
            dataset.query(f'PriceArea == "{pricearea}"').to_csv(self.SAVE_DIR / f'{pricearea}.csv')

        dataset.to_csv(self.SAVE_DIR / f'AllPriceAreas.csv')
        actual_wind_power.to_csv(self.SAVE_DIR / 'actual_wind_power.csv')
        balancing_prices.to_csv(self.SAVE_DIR / 'balancing_prices.csv')
        day_ahead_prices.to_csv(self.SAVE_DIR / 'day_ahead_prices.csv')
        avg_weather.to_csv(self.SAVE_DIR / f'avg_weather_{weather_municipality}.csv')

if __name__ == '__main__':
    
    # Set path to data and save folder
    DATA_DIR    = Path('../../../data/assignment1')
    SAVE_DIR    = Path(r'../../../data/assignment1/processed')
    
    # Define dataset processor object
    processor   = DataProcessor(DATA_DIR=DATA_DIR, SAVE_DIR=SAVE_DIR)

    # Load and combine data sources
    processor.combine_and_save_data_sources(weather_municipality='Roskilde', priceareas=['DK1', 'DK2'])

    # Load DK1 and DK2 datasets
    DK1_dataset = pd.read_csv(SAVE_DIR / 'DK1.csv', index_col=0)
    DK2_dataset = pd.read_csv(SAVE_DIR / 'DK2.csv', index_col=0)

    # Check if missing data occurs
    print(f"NaN values occuring in DK1 dataset? {DK1_dataset.isna().any().any()}")
    print(f"NaN values occuring in DK2 dataset? {DK2_dataset.isna().any().any()}")