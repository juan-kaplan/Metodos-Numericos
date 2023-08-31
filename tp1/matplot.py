import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.interpolate import lagrange, CubicSpline

def create_line_graph(x_datapoints, y_datapoints, functions):
    fig, axs = plt.subplots()
    axs.plot(x_datapoints, y_datapoints, 'o', label='Data Points')
    for function in functions:
        axs.plot(function[0], function[1], '-', label= function[2])
  
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()
    return axs

def create_error_graph(functions):
    fig, axs = plt.subplots()
    for function in functions:
        axs.plot(function[0], function[1], '-', label= function[2])
  
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()
    axs.grid(True)
    return axs

def second_derivative(func, var):
    x = sp.symbols(var)
    first_derivative = sp.diff(func, x)
    second_derivative = sp.diff(first_derivative, x)
    return second_derivative

def find_second_derivative_zeros(func, var):
    x = sp.symbols(var)
    second_derivative = sp.diff(func, x, 2)
    critical_points = sp.solve(second_derivative, x)
    return critical_points

def my_function(x):
    return 0.05 ** sp.abs(x) * sp.sin(5 * x) + sp.tanh(2 * x) + 2

x_points2 = np.array(find_second_derivative_zeros(sp.sin(2 * sp.symbols('x')) + sp.symbols('x')**2, "x"))
y_points2 = my_function(x_points2)

x_interval = np.arange(-3, 3.05, 0.05)
x2 = np.array(x_points2)
y2 = np.array(y_points2)

poly_lagrange2 = lagrange(x2, y2)
poly_spline2 = CubicSpline(x2, y2)

y_interp_lagrange2 = poly_lagrange2(x_interval)
y_interp_splines2 = poly_spline2(x_interval)

create_line_graph(x_points2, y_points2, [(x_interval, y_interp_splines2, "Cubic Spline")])
plt.show()