import math

import pandas

from FetchData import FetchData


def main():
    fetch_data = FetchData()
    Running_plan = fetch_data.get_data()
    pandas.set_option('display.max_columns', None)
    print(Running_plan.head(0))
    sum = 0;
    for i in range(Running_plan.shape[0]):
        number = Running_plan.iloc[i, 2]
        if not math.isnan(number):
            roundednumber = int(round(number))
            sum += roundednumber
    print(sum)

    datelist = []
    for i in range(Running_plan.shape[0]):
        s = Running_plan.iloc[i, 1]
        if datelist.count(s) == 0:
            datelist.append(s)
    print(sum/len(datelist))


main()
