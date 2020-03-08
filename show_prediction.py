import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/submission_6_lorien.csv", sep=";").sort_values(by='timestamp', ascending=True)

data.plot(x='timestamp', y='energy', kind='line')
plt.show()
