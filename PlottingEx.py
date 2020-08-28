import numpy as np
import matplotlib.pyplot as plt

# Sin(X) Graph
# x = np.linspace(-1, 10, 50)
# print(x)
# y = np.sin(x)
# print(y)
# plt.figure()
# plt.plot(x, y, '*-b') #(x,y, style)
# plt.scatter(x, y**2, c='red')
# plt.xlabel("x")
# plt.xlabel("y")
# plt.title("sin(x) and sin(x**2)")
# plt.show()

# Exponential Graph
# x = np.logspace(0., 3., 10) # 10^0 to 10^3
# y = np.exp(x**0.2+10.0*np.tanh(0.5*np.log(x)))
# plt.figure()
# plt.loglog(x, y, marker="d", linestyle='-.') # semilogx, semilogy also exist
# plt.xlabel("Frequency")
# plt.ylabel("Gain")
# plt.title("Frequency vs Gain")
# plt.show()

#Scatter Plot
plt.figure()
for color in ['blue', 'orange', 'green']:
    n = 60
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    plt.scatter(x, y, c=color, s=scale, label=color,
                alpha=0.3, edgecolors='none')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Random Circles")
plt.show()