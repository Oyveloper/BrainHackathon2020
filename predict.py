from Net import get_trained_net as net
import torch
import FetchData
from FetchData import FetchData
import pandas
from utils import *


global turbinelist
global weatherForecast


def predict_48(from_time):
    global weatherForecast
    weatherForecast = FetchData.get_weather_forecast()
    datelist = get48hourdatetimelist(from_time)
    for i in range(47):
        if i > 10:
            turbinelist.append("T" + str(i + 1))
        else:
            turbinelist.append("T0" + str(i + 1))

    for i in range(48):
        predict_hour(datelist[i])
    pass


def predict_hour(time):
    sum = 0
    for t in turbinelist:
        sum += predict_turbine(time, t)
    return sum


def predict_turbine(time, turbine):
    net.predict(turbine, data)


def getWeatherData(time):
    global weatherForecast
    for i in range(weatherForecast.shape(0)):
        if weatherForecast.iloc(i,0) == "WP" and weatherForecast.iloc(i,1) == time:
            return [weatherForecast.iloc()]



def get48hourdatetimelist(start_time):
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

    for i in range(48):
        hour = hour + 1
        if (hour == 24):
            hour = 0
            numsplit[0] = numsplit[0] + 1
            if numsplit[0] > 31:
                numsplit[0] = 0
                numsplit[1] = numsplit[1] + 1
        datelist.append(str(numsplit[0]) + "." + str(numsplit[1]) + "." + str(numsplit[2]) + " " + str(hour) + ":00")
    return datelist


net = get_trained_net()
input = torch.tensor([1, 100, 1000])

predict = net(input)
