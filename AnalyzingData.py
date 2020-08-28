import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create some data
months = pd.date_range(start='20190101', periods=12, freq='M')
change = np.random.normal(0,1.2,(12, 3))
stocks = ['GOOG', 'TSLA', 'APPL']
df = pd.DataFrame(change, index=months, columns=stocks)

print(df.head(3), '\n')
print(df.tail(2), '\n')
print(df.describe(), '\n')

## Selection using labels

# One columns
print(df['GOOG'], '\n')

# A slice of rows
print(df[2:5], '\n')

# Multiple rows & columns
# Endpoints INCLUDED, unlike in regular Python slicing syntax
print(df.loc['2019-07-31':'2019-09-30',['TSLA','GOOG']], '\n')

## Selection using conditions

print(df, '\n')
print(df.loc[df['GOOG'] > 2.5,:], '\n')         # Some rows
print(df.loc[df.index >= '2019-08-15',:], '\n') # Some rows
print(df[df > 0.5], '\n')                       # All data

