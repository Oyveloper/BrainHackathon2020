from Net import get_trained_net
import torch
import FetchData
from FetchData import FetchData
import pandas
from utils import *
from tqdm import tqdm


turbinelist = []
weatherForecast = []
runningPlan = []
net = get_trained_net()

fetchData = FetchData()


def getData(time, t):
    global weatherForecast
    global runningPlan
    pwlim = 0
    state = 0
    for i, row in runningPlan.iterrows():
        if runningPlan.iloc[i,0] == t:
            pwlim = runningPlan.iloc[i,2]
            state = runningPlan.iloc[i, 3]
    for i, row in weatherForecast.iterrows():
        if weatherForecast.iloc[i,0] == "WP" and weatherForecast.iloc[i,1] == time:
            return [turbine_to_number(t),weatherForecast.iloc[i,:]['SUB_WIND_SPEED_110'],weatherForecast[i,:]["SUB_WIND_DIR_110"],pwlim,weatherForecast[i,:]["SUB_AIR_TEMP"],state]


def predict_48(from_time):
    global weatherForecast
    weatherForecast = fetchData.get_weather_forecast()
    global runningPlan
    runningPlan = fetchData.get_running_plan()
    datelist = get48hourdatetimelist(from_time)

    total_list = []

    for i in range(47):
        if i > 10:
            turbinelist.append("T" + str(i + 1))
        else:
            turbinelist.append("T0" + str(i + 1))

    for i in tqdm(range(48)):
        predicted = predict_hour(datelist[i])
        total_list.append([datelist[i], predicted])

    return total_list


def predict_hour(time):
    sum = 0
    for t in turbinelist:
        sum += predict_turbine(time, t)
    return sum


def predict_turbine(time, turbine):
    data = getData(time, turbine)
    return net(torch.tensor(data)).item()





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

