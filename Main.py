import pandas

from FetchData import FetchData


def main():
    fetch_data = FetchData()
    Running_plan = fetch_data.get_weather_forecast()
    pandas.set_option('display.max_columns', None)
    print(Running_plan.head(10))







main()