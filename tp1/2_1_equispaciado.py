import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

def my_function(x):
    return 0.05 ** np.abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

x_points = [-3, -2, -1, 0, 1, 2, 3]
y_points = [my_function(x) for x in x_points]

# Sample data points
x = np.array(x_points)
y = np.array(y_points)

# Compute the Lagrange polynomial
poly = lagrange(x, y)

#Interval
x_interval = np.arange(-3, 3.05, 0.05)

#print(x_interval)

# Evaluate the polynomial at some points
y_interp = poly(x_interval)

#Original function
y_original = my_function(x_interval)

# Plot the original data points and the interpolated polynomial
plt.subplot(2, 1, 1)
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_interval, y_original, '-', label= 'func original')
plt.plot(x_interval, y_interp, '-', label='Interpolated Polynomial')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Add vertical lines between the original function and the interpolated polynomial
for i in range(0, len(x_interval), 3):
    plt.vlines(x_interval[i], y_original[i], y_interp[i], colors='r', linestyles='dotted')

error_p = [round(abs(y_original[i] - y_interp[i]), 4) for i in range(0, len(x_interval), 1)]

plt.subplot(2, 1, 2)
plt.plot(x_interval, error_p, '-', label='error points')
plt.xlabel('X')
plt.ylabel('Y')

plt.legend()
plt.grid(True)
plt.show()






#2 funcion a

#f(x)  / x (2 decimales)
# 0.99  -3
# 1.00  -2
# 1.08  -1
# 2.00   0
# 2.92   1
# 3.00   2
# 3.00   3

