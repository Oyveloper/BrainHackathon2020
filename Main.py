from FetchData import FetchData


def main():
    fetch_data = FetchData()
    Running_plan = fetch_data.get_running_plan()
    print(Running_plan)







main()