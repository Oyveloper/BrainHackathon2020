import pandas as pd
from predict import Predictor

def listTo(predList):
    df = pd.DataFrame(predList, columns=["Timestamp", "Energy"])
    df.to_csv('data/testList.csv', index=False, header=True)

if __name__ == "__main__":
    predictor = Predictor()
    listTo(predictor.predict_48('2019-01-06 00:00:00+00:00'))


