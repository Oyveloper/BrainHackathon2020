import pandas as pd
from predict import Predictor

def listTo(predList):
    df = pd.DataFrame(predList, columns=["Timestamp", "Energy"])
    df.to_csv('data/testList.csv', index=False, header=True)

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict_48('06.01.2019 00:00')


