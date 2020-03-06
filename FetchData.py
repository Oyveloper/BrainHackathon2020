from typing import Any, Union

import pandas as pd
import csv

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from data import *


class FetchData:
    """From data fetch all shit and sort it in a good way"""

    def __inti__(self):
        pass

    def sort_data(self):
        pass

    def get_running_plan(self):
        return pd.read_csv("./data/running_plan.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
        #return self.running_plan

    def get_data(self):
        return pd.read_csv("./data/data.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
        #return self.data

    def get_weather_forecast(self):
        return pd.read_csv("./data/weather_forecast_utc.csv")
        #return self.weather_forecast

#running_plan = pd.read_csv("./data/running_plan.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
#print(running_plan.head())
