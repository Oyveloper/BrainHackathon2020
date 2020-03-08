import pandas as pd
from predict import Predictor

def listTo(predList):
    round = 6
    df = pd.DataFrame(predList, columns=["timestamp", "energy"])
    df.to_csv(f"data/submission_{round}_lorien.csv", index=False, header=True, sep=";")

if __name__ == "__main__":
    predictor = Predictor()
    listTo(predictor.predict_48('2019-04-23 00:00:00+00:00'))


