import math

import pandas

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


    for t in turbinelist:
        sumPerTurbine = 0
        for i in range(Running_plan.shape[0]):
            if Running_plan.iloc[i, 0] == t:
                sumPerTurbine += Running_plan.iloc[i, 2]
        print("Turbine: " + t + "\tProduced average of: " + str(sumPerTurbine))


def main():
    fetch_data = FetchData()
    Running_plan = fetch_data.get_data()
    pandas.set_option('display.max_columns', None)
    print(Running_plan.head(0))
    educatedguess(Running_plan)


main()
