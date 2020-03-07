import math

import pandas
import datetime
import re
from FetchData import FetchData



def educatedguess(Running_plan):
    sum = 0;
    for i in range(Running_plan.shape[0]):
        number = Running_plan.iloc[i, 2]
        if not math.isnan(number):
            roundednumber = int(round(number))
            sum += roundednumber
    print("Sum of all power generated for all hour for all turbines is: " + str(sum))

    datelist = []
    for i in range(Running_plan.shape[0]):
        s = Running_plan.iloc[i, 1]
        if datelist.count(s) == 0:
            datelist.append(s)
    eachHourProduce = sum / len(datelist)
    print("Sum of power generated for all turbines each hour: " + str(eachHourProduce))

    turbinelist = []
    for i in range(Running_plan.shape[0]):
        turb = Running_plan.iloc[i, 0]
        if turbinelist.count(turb) == 0:
            turbinelist.append(turb)
    eachHourEachTurbine = eachHourProduce / len(turbinelist)
    print("Average power each hour each turbine: " + str(eachHourEachTurbine))

    # 31.10.2018 00:00 UTC
    datelist = []
    start = "17.12.2018 00:00"
    start = start.replace(" ", ".")
    split = start.split(".")
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

    twodaylist = []
    for i in range(48):
        twodaylist.append(eachHourProduce)
    donelist = []
    for i in range(48):
        donelist.append(datelist[i] + ";" + str(twodaylist[i]))
        print(datelist[i] + ";" + str(twodaylist[i]))
    hei = pandas.DataFrame.to_csv(donelist)
    print(hei)

    for d in datelist:
        dateaverage = 0
        for i in range(Running_plan.shape[0]):
            if Running_plan.iloc[i, 1] == d:
                num = Running_plan.iloc[i, 2]
                if not math.isnan(num):
                    dateaverage += int(round(num))
        print("Date: " + d + "\tAverage: " + str(dateaverage))

    for t in turbinelist:
        sumPerTurbine = 0
        for i in range(Running_plan.shape[0]):
            if Running_plan.iloc[i, 0] == t:
                num = Running_plan.iloc[i, 2]
                if not math.isnan(num):
                    sumPerTurbine += int(round(num))
        print("Turbine: " + t + "\tProduced average of: " + str(sumPerTurbine))


def main():
    fetch_data = FetchData()
    Running_plan = fetch_data.get_data()
    educatedguess(Running_plan)
    pandas.set_option('display.max_columns', None)
    print(Running_plan.head(0))
    educatedguess(Running_plan)


main()

