import numpy as np
import matplotlib.pyplot as plt

# # Stock simulation 1
# T = 100
# stock = np.zeros(T)
# mu = 1.2
# std = 5.0

# for t in range(1,T):
#     stock[t] = stock[t-1] + np.random.normal(mu,std)

# plt.plot(stock)
# plt.xlabel('Day')
# plt.ylabel('APPL ($)')
# plt.show()

# Stock simulation 2

T = 100
n = 10
stock = np.zeros((n,T))

for t in range(1,T):
    mu  = 1.2 * np.exp(-t/10)
    std = 0.25 * (1 + np.sin(t))
    stock[:,t] = stock[:,t-1] + np.random.normal(mu,std,n)

plt.plot(stock.transpose())
plt.xlabel('Day')
plt.ylabel('APPL ($)')
plt.show()