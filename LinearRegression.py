import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

#Supervised Learning Example

from sklearn import linear_model

N = 10
x = 1.0 + np.random.uniform(size=N)
y = 20.0 - 10.0 * x + 2.3 * x**2 + 0.2 * np.random.normal(size=N)

# Learn
model = linear_model.LinearRegression(fit_intercept=True) #has by default intercepts=true, can remove 
X = x.reshape((-1,1)) #need to reshape bc of the different sizes of the matrix's and vectors 
model.fit(X, y) #training the model 

# Predict
xx = np.linspace(1, 2, 10)
XX = xx.reshape((-1,1))
yy = model.predict(XX)

# Plot
plt.figure()
plt.scatter(x,y,c='r',label="Data points")
plt.plot(xx,yy,"-*b",label="Predictions")
plt.xlabel("Height [m]")
plt.ylabel("Hair thickness [mm]")
plt.title("Linear Regression Example")
plt.legend()
plt.show()

print(model.intercept_)
print(model.coef_)