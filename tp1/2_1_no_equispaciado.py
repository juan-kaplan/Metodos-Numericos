import numpy as np
from scipy.interpolate import lagrange
from scipy.misc import derivative
import matplotlib.pyplot as plt

def my_function(x):
    return 0.05 ** np.abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

def get_x_points2(f, start, end):
    x_points = [start]
    x = start
    while x <= end:
        if (abs(derivative(f, x) - derivative(f,x_points[-1])) >= 0.2):
            x_points.append(x)
        x += 0.1
    x_points.append(end)
    return x_points

x_points = np.union1d(get_x_points2(my_function, -1.4, 1.4), np.array(np.arange(-3, 3.5, 0.5)))
y_points = [my_function(x) for x in x_points]
print(x_points)
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
plt.show()