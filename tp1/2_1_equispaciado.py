import numpy as np
import numpy
from scipy.interpolate import lagrange, CubicSpline
import matplotlib.pyplot as plt
from scipy.misc import derivative

def get_x_points2(f):
    k = np.arange(1, 5+1)
    nodes = 0.5 * (-3 + 3) + 0.5 * (3 - -3) * np.cos((2 * k - 1) * np.pi / (2 * 5))
    nodes_purged = nodes[np.abs(nodes) > 1]
    x_points = [-1]
    x = -1
    while x <= 1:
        if np.sign(derivative(f, x, dx=1e-6)) != np.sign(derivative(f, x + 0.001, dx=1e-6)):
            x_points.append(x)
        x += 0.001
    x_points.append(1)
    x_points.extend(nodes_purged)
    x_points.sort()
    return x_points 

def chebyshev_nodes(a, b, n):
    k = np.arange(1, n+1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return nodes

def my_function(x):
    return 0.05 ** np.abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

def create_line_graph(axs, x_datapoints, y_datapoints, functions, title = ""):
    axs.plot(x_datapoints, y_datapoints, 'o', label='Data Points')
    for function in functions:
        axs.plot(function[0], function[1], '-', label= function[2])
  
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()
    return axs

def create_error_graph(axs, functions, title=""):
    for function in functions:
        axs.plot(function[0], function[1], '-', label= function[2])

    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()
    axs.grid(True)
    return axs


# Points with x space between them
x_points = np.array([-3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3.0])
y_points = my_function(x_points)

# Compute the Interpolation polynomial
poly_lagrange = lagrange(x_points, y_points)
poly_spline = CubicSpline(x_points, y_points)

# Interval
x_interval = np.arange(-3, 3.05, 0.05)

# Evaluate the polynomial at some points
y_interp_lagrange = poly_lagrange(x_interval)
y_interp_splines = poly_spline(x_interval)

# Original function
y_original = my_function(x_interval)

# Create a figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot the original data points and the interpolated polynomial
create_line_graph(axs[0], x_points, y_points, [(x_interval, y_original, "func original"), (x_interval, y_interp_lagrange, "Lagrange "), (x_interval, y_interp_splines, "Cubic Spline")])
axs[0].set_title("Interpolacion equiespaciada")

# Unevenly spaced points
# x_points2 = np.flip(chebyshev_nodes(-3, 3, 15))
x_points2 = np.array(get_x_points2(my_function))
y_points2 = my_function(x_points2)

x2 = np.array(x_points2)
y2 = np.array(y_points2)

poly_lagrange2 = lagrange(x2, y2)
poly_spline2 = CubicSpline(x2, y2)

y_interp_lagrange2 = poly_lagrange2(x_interval)
y_interp_splines2 = poly_spline2(x_interval)

create_line_graph(axs[1], x2, y2, [(x_interval, y_original, "func original"), (x_interval, y_interp_lagrange2, "Lagrange"),   (x_interval, y_interp_splines2, "Cubic Spline")])
axs[1].set_title("Interpolacion no equiespaciada")

# Calculate error points
error_lagrange1 = np.abs(y_original - y_interp_lagrange)
error_lagrange2 = np.abs(y_original - y_interp_lagrange2)
error_spline1 = np.abs(y_original - y_interp_splines)
error_spline2 = np.abs(y_original - y_interp_splines2)
# Plot the error points

create_error_graph(axs[2], [(x_interval, error_lagrange1, "Lagrange equispaciado error"), (x_interval, error_lagrange2, "Lagrange no-equispaciado error"), (x_interval, error_spline1, "Spline equispaciado error"), (x_interval, error_spline2, "Spline no-equispaciado error")])
axs[2].set_title("Error de polinomios")

plt.tight_layout()
axs[2].set_ylim(0.5, 3.5)
plt.show()
