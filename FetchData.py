from typing import Any, Union

import pandas as pd
import csv

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from data import *
import utils


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


    def get_column_num(self):
        return len(self.columns)
    def sort_data(self):
        pass

    def get_running_plan(self):
        return pd.read_csv("./data/running_plan.2019-02-21.a11718a4-5e2e-11ea-815b-000d3a64d565.csv")
        #return self.running_plan

    def get_data(self):
        data = pd.read_csv("./data/data.2019-02-21.a11718a4-5e2e-11ea-815b-000d3a64d565.csv")
        data.fillna(method='ffill', inplace=True)
        data['turbine'] = data['turbine'].apply(lambda x: utils.turbine_to_number(x))
        return data
        


    def get_training_data(self):
        """
        Gets training data 
        Returned as a tuple of (train_X, train_Y)
        """
        data = self.get_data()
        # Clean up the data 
        division_point = int(len(data) * self.train_split)
        train_data = data.iloc[:division_point, :]
        train_X = train_data[self.columns]
        train_Y = train_data['ActivePower (Average)']

        return (train_X, train_Y)


    def get_testing_data(self):
        """
        Gets testing data 
        Returned as a tuple of (train_X, train_Y)
        """
        # get raw data and split
        data = self.get_data()
        division_point = int(len(data) * self.train_split)
        test_data = data.iloc[division_point:, :]

        # separate to x and y 
        test_X = test_data[self.columns]
        test_Y = test_data['ActivePower (Average)']
        return (test_X, test_Y)
        

    def get_weather_forecast(self):
        return pd.read_csv("./data/weather_forecast_utc.csv")
        #return self.weather_forecast

#running_plan = pd.read_csv("./data/running_plan.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
#print(running_plan.head())
