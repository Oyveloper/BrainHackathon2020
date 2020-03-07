from Net import get_trained_net, dataFetcher
import torch
import FetchData
from FetchData import FetchData
import pandas
from utils import *
from tqdm import tqdm
import numpy as np
import re

from datetime import datetime, timedelta


class Predictor():
    def __init__(self):
        dataFetcher = FetchData()
        self.dataFetcher = dataFetcher
        self.running_plan = self.dataFetcher.get_running_plan()
        self.weather_data = self.dataFetcher.get_weather_forecast()
        # Filter for whole park only 
        self.weather_data = self.weather_data[self.weather_data['windpark_zone'] == 'WP']
        self.turbines = set(self.running_plan.turbine.values)

        self.net = get_trained_net()

        self.date_format = "%Y-%m-%d %H:%M:%S%z"
        



    def predict_48(self, start_time_str):
        predlist = []
        start_date_time = datetime.strptime(start_time_str, self.date_format)
        for i in tqdm(range(48)):
            current_time = start_date_time + timedelta(hours=i)
            current_time_stamp = current_time.strftime(self.date_format[:-2]) + "+00:00"
            predlist.append([current_time_stamp, self.predict_hour(current_time_stamp)])
        return predlist



    def predict_hour(self, timestamp):


        weather = self.weather_data[self.weather_data['datetime_start_utc'] == timestamp]
        weather.sort_values(by=['datetime_forecast_utc'], inplace=True, ascending=False)
        weather = weather[['SUB_WIND_SPEED_110', 'SUB_WIND_DIR_110', 'SUB_AIR_TEMP_2']]
        weather = weather.values[0]

        total = 0
        
        for turbine in self.turbines:
            turbine_id = turbine_to_number(turbine)
            turbine_data = self.running_plan[self.running_plan['turbine'] == turbine]
            turbine_data = turbine_data[turbine_data['timestamp'] == timestamp][['ActivePowerLimit', 'StateRun']].values[0]

          
            predict_input_list = [turbine_id] + weather.tolist() + turbine_data.tolist()
            predict_input = torch.tensor(predict_input_list)

            total += self.net(predict_input).item()
        return total

    def get48hourdatetimelist(self, start_time):
        datelist = []
        start_time = start_time.replace(" ", ".")
        split = start_time.split(".")
        numsplit = []
        print(split)
        hourandminute = split[3].split(":")
        hour = int(hourandminute[0])
        for i in range(len(split)):
            if i != 3:
                numsplit.append(int(split[i]))
        year = numsplit[0]
        numsplit[0] = numsplit[2]
        numsplit[2] = year
        for i in range(48):
            if (hour == 24):
                hour = 0
                numsplit[0] = numsplit[0] + 1
                if numsplit[1] % 2 == 0:
                    if numsplit[0] > 30:
                        numsplit[0] = 0
                        numsplit[1] = numsplit[1] + 1
                else:
                    if numsplit[0] > 31:
                        numsplit[0] = 0
                        numsplit[1] = numsplit[1] + 1
            day = numsplit[0]
            numsplit[0] = numsplit[2]
            numsplit[2] = day
            datelist.append(
                str(numsplit[0]) + "." + str(numsplit[1]) + "." + str(numsplit[2]) + " " + str(hour) + ":00")
            hour = hour + 1
            return datelist
            

if __name__ == "__main__":
    predictor = Predictor()
    print(predictor.predict_hour('2019-01-05 00:00:00+00:00'))


