from typing import Any, Union

import pandas as pd
import csv

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from data import *
import utils

from tqdm import tqdm

import torch

import os


TRAIN_X_PATH = "./model/train_X.pth"
TRAIN_Y_PATH = "./model/train_Y.pth"
TEST_X_PATH = "./model/test_X.pth"
TEST_Y_PATH = "./model/test_Y.pth"

class FetchData:
    """From data fetch all shit and sort it in a good way"""

    def __init__(self):
        self.train_split = 0.8
        self.columns = ['turbine',
                        'WindSpeed (Average)',
                        'WindDirection (Average)',
                        'AmbientTemp (Average)',
                        'ActivePowerLimit (End)',
                        'StateRun (End)']
        self.data = None


    def get_inputsize(self):
        return self.load_testing_data()[0].size()[1]
    def sort_data(self):
        pass

    def get_running_plan(self):
        return pd.read_csv("./data/running_plan.2019-04-23.a45c8efe-5e2e-11ea-b199-000d3a64d565.csv")
        #return self.running_plan

    def get_data(self):
        if self.data is None:


            print("loading all datafiles")
            for filename in tqdm(os.listdir("./data")):
                if filename.startswith("data."):
                    sub_data = pd.read_csv(f"./data/{filename}")
                    if self.data is None:
                        self.data = sub_data
                    else:
                        self.data = self.data.append(sub_data, ignore_index=True)

            

            self.data.fillna(method='ffill', inplace=True)
            self.data['turbine'] = self.data['turbine'].apply(lambda x: utils.turbine_to_number(x))
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            return self.data
        else:
            return self.data


    def load_training_data(self):
        train_X = torch.load(TRAIN_X_PATH)
        train_Y = torch.load(TRAIN_Y_PATH)

        return (train_X, train_Y)
    
    def load_testing_data(self):
        test_X = torch.load(TEST_X_PATH)
        test_Y = torch.load(TEST_Y_PATH)

        return (test_X, test_Y)
        


    def build_training_and_loading_data(self):
        print("building training data")
        train_X, train_Y = self.get_training_data()
        torch.save(train_X, TRAIN_X_PATH)
        torch.save(train_Y, TRAIN_Y_PATH)

        print("building testing data")
        test_X, test_Y = self.get_testing_data()
        torch.save(test_X, TEST_X_PATH)
        torch.save(test_Y, TEST_Y_PATH)

    def get_training_data(self):
        """
        Gets training data 
        Returned as a tuple of (train_X, train_Y)
        """
        data = self.get_data()

        # Clean up the data 
        division_point = int(len(data) * self.train_split)
        train_data = data.iloc[:division_point, :]

        # Get overlapping weather data
        weather_data = self.get_weather_forecast()
        weather_data = weather_data[weather_data['windpark_zone'] == "WP"]
        weather_data = weather_data[weather_data['datetime_start_utc'].isin(train_data['timestamp'])]


        train_X = []
        train_Y = []

        for index, row in tqdm(train_data.iterrows()):
            timestamp = row.timestamp
            weather = weather_data[weather_data['datetime_start_utc'] == timestamp]
            weather.sort_values(by=['datetime_forecast_utc'], inplace=True, ascending=False)


            weather = weather.select_dtypes(include=["float64"]).values[0]
            
            train_X.append(row[self.columns].values.tolist() + weather.tolist())
            
        # train_X = train_data[self.columns] 
        train_Y = train_data['ActivePower (Average)']

        return (torch.tensor(train_X), torch.tensor(train_Y.values).view(-1, 1))


    def get_testing_data(self):
        """
        Gets testing data 
        Returned as a tuple of (train_X, train_Y)
        """
        # get raw data and split
        data = self.get_data()
        division_point = int(len(data) * self.train_split)
        test_data = data.iloc[division_point:, :]

        weather_data = self.get_weather_forecast()
        weather_data = weather_data[weather_data['windpark_zone'] == "WP"]
        weather_data = weather_data[weather_data['datetime_start_utc'].isin(test_data['timestamp'])]


        test_X = []
        test_Y = []


        weather_data.sort_values(by=['datetime_forecast_utc'], inplace=True, ascending=False)
        weather_data.drop_duplicates(subset='datetime_forecast_utc', inplace=True, keep='first')

        for index, row in tqdm(test_data.iterrows()):
            timestamp = row.timestamp
            weather = weather_data[weather_data['datetime_start_utc'] == timestamp]
            weather.sort_values(by=['datetime_forecast_utc'], inplace=True, ascending=False)

            weather = weather.select_dtypes(include=["float64"]).values[0]
            
            test_X.append(row[self.columns].values.tolist() + weather.tolist())

        # separate to x and y 

        test_Y = test_data['ActivePower (Average)']

        
        return (torch.tensor(test_X), torch.tensor(test_Y.values).view(-1, 1))
        

    def get_weather_forecast(self):
        return pd.read_csv("./data/weather_forecast_utc.csv")
        #return self.weather_forecast

#running_plan = pd.read_csv("./data/running_plan.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
#print(running_plan.head())
