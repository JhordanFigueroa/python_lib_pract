import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

# Load data from NYT dataset
df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv",parse_dates=[0])
df.head(5)

# Pivot in states
cases_states = df.pivot(index='date',columns='state',values='cases').fillna(0)

# Compute daily change and smooth accross 7 days
cases_states = cases_states.diff().fillna(0)

# Extract CA data
x = (cases_states.index - pd.to_datetime('2020-01-21')).total_seconds().to_numpy() / (3600.0 * 24.0)
y = cases_states['New York'].to_numpy() #transfors into a numpy array just makes an array - don't really need it but its to be consistent 

# # Plot Before Training Model
# plt.plot(x, y)
# plt.xlabel('Days since 2020-01-21')
# plt.ylabel('Cases per day (7-days average)')
# plt.show()

from sklearn import linear_model

# Create train & test sets
start = 50
end = 130
x_train      = x[start:end].reshape((-1,1))
x_test       = x[end:].reshape((-1,1))
y_train      = y[start:end]
y_test       = y[end:]

# Fit
model = linear_model.PoissonRegressor()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Plot
plt.figure()
plt.plot(x_train,y_train,'-r',label="Training set")
plt.plot(x_test,y_pred,'-b',label="Predictions")
plt.plot(x_test,y_test,'-g',label="Truth")
plt.xlabel("Days")
plt.ylabel("Number of death")
plt.legend()
plt.title("New York COVID-19 data (")
plt.show()