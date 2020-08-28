import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv",parse_dates=[0])
print(df)

#First Plot
cases_states = df.pivot(index='date',columns='state',values='cases')
cases_states = cases_states.fillna(0)
# print(cases_states)
# cases_states.plot(y=['California','New York','Florida'])

#Second Plot
daily_cases_states = cases_states.diff()
daily_cases_states = daily_cases_states.fillna(0)
print(daily_cases_states)
daily_cases_states.plot(y=['California','New York','Florida'])