import numpy as np
from scipy.interpolate import lagrange, CubicSpline, RectBivariateSpline
import matplotlib.pyplot as plt
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def chebyshev_nodes(a, b, n):
    k = np.arange(1, n+1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return nodes

def calculate_mass_error(y_original, y_interpolated):
    return sum(np.abs(y_original[::2] - y_interpolated[::2]))

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

#Graph with 10 nodes
x_points = np.linspace(-3, 3, 13)
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
x_points2 = np.flip(chebyshev_nodes(-3, 3, 10))
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
plt.show()

# Calculate error with different number of nodes
def calculate_error_graph_nodes(a, b):
    fig, axs = plt.subplots()
    nodes_error_lagrange = {}
    nodes_error_splines = {}

    for i in range(a, b + 1):
        x_points = np.linspace(-3, 3, i)
        y_points = my_function(x_points)
        x_interval = np.arange(-3, 3.05, 0.05)
        poly_lagrange = lagrange(x_points, y_points)
        poly_spline = CubicSpline(x_points, y_points)

        y_interp_lagrange = poly_lagrange(x_interval)
        y_interp_splines = poly_spline(x_interval)

        nodes_error_lagrange[i] = calculate_mass_error(y_original, y_interp_lagrange)
        nodes_error_splines[i] = calculate_mass_error(y_original, y_interp_splines)

    print(nodes_error_lagrange)
    axs.plot(list(nodes_error_lagrange.keys()), nodes_error_lagrange.values(), "-", label = "Error Lagrange")
    axs.plot(list(nodes_error_splines.keys()), nodes_error_splines.values(), "-", label = "Error Splines")
    axs.set_xlabel('Nodes')
    axs.set_ylabel('Mass error')
    axs.set_title("Mass error for different nodes")
    axs.legend()

    return axs

error_graph_x_space = calculate_error_graph_nodes(5, 13)

#3D Function
# Define the function f2(x1, x2)
def f2(x1, x2):
    term1 = 0.7 * np.exp(-((9*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)
    term2 = 0.45 * np.exp(-((9*x1 + 1)**2)/9 - ((9*x2 + 1)**2)/5)
    term3 = 0.55 * np.exp(-((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4)
    term4 = -0.01 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    
    return term1 + term2 + term3 + term4

# Create a meshgrid for x1 and x2
x1_vals = np.linspace(-1, 1, 100)
x2_vals = np.linspace(-1, 1, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)

# Calculate function values for each combination of x1 and x2
f_values = np.array([[f2(x1, x2) for x1 in x1_vals] for x2 in x2_vals])

# Perform cubic spline interpolation with 10 points for each variable
x1_interp = np.linspace(-1, 1, 20)
x2_interp = np.linspace(-1, 1, 20)

# Create a grid for interpolation
X_interp, Y_interp = np.meshgrid(x1_interp, x2_interp)

# Calculate function values for the interpolation grid
f_values_interp = np.array([[f2(x1, x2) for x1 in x1_interp] for x2 in x2_interp])

# Create a bivariate spline for the interpolated function
spline = RectBivariateSpline(x1_interp, x2_interp, f_values_interp)

# Evaluate the spline on the interpolation grid
Z_interp = spline(x1_interp, x2_interp)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

# Plot the surface of the original function
axs[0].plot_surface(x1_mesh, x2_mesh, f_values, cmap='viridis')
axs[0].set_xlabel('x1')
axs[0].set_ylabel('x2')
axs[0].set_zlabel('f2(x1, x2)')
axs[0].set_title('Original Function f2(x1, x2)')

# Plot the surface of the interpolated function
axs[1].plot_surface(X_interp, Y_interp, Z_interp, cmap='viridis')
axs[1].set_xlabel('x1')
axs[1].set_ylabel('x2')
axs[1].set_zlabel('f2(x1, x2)')
axs[1].set_title('Interpolated Function with RectBivariateSpline')

plt.tight_layout()
plt.show()


