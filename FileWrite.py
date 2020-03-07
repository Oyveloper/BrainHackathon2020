import pandas as pd
from predict import predict_48

def listTo(predList):
    df = pd.DataFrame(predList, columns=["Date", "Energy"])
    df.to_csv('data/testList.csv', index=False, header=True)

if __name__ == "__main__":
    print("hei")
    listTo(predict_48("17.12.2018 00:00"))