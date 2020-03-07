import pandas as pd


def listTo(predList):
    df = pd.DataFrame(predList, columns=["Date", "Energy"])
    df.to_csv('data/testList.csv', index=False, header=True)
