from FetchData import FetchData


def main():
    fetch_data = FetchData()
    Running_plan = fetch_data.get_weather_forecast()
    print(Running_plan.da)







main()