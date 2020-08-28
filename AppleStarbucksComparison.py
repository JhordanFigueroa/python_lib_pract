import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('https://web.stanford.edu/~lcambier/pc/stocks.csv',parse_dates=[1])

df2 = df.pivot(index='Date',columns='Stock',values='Open')
df2.plot(y=['APPL','SBUX'])
plt.show()