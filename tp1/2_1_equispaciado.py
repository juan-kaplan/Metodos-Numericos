import numpy as np
from scipy.interpolate import lagrange, CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def chebyshev_nodes(a, b, n):
    k = np.arange(1, n+1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return nodes[::-1]

def calculate_mass_error(y_original, y_interpolated):
    return sum(np.abs(y_original[::2] - y_interpolated[::2]))

def my_function(x):
    return 0.05 ** np.abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

def create_line_graph(axs, x_datapoints, y_datapoints, functions):
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

# Grafico con 10 nodos
x_points = np.linspace(-3, 3, 13)
y_points = my_function(x_points)

# Computo metodos lagrange y cubic spline en los nodos
poly_lagrange = lagrange(x_points, y_points)
poly_spline = CubicSpline(x_points, y_points)

# Intervalo donde se va a evaluar la interpolacion
x_interval = np.arange(-3, 3.05, 0.05)

y_interp_lagrange = poly_lagrange(x_interval)
y_interp_splines = poly_spline(x_interval)

# Funcion original (GROUND TRUTH)
y_original = my_function(x_interval)

# Creacion de grafico
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
create_line_graph(axs[0], x_points, y_points, [(x_interval, y_original, "func original"), (x_interval, y_interp_lagrange, "Lagrange "), (x_interval, y_interp_splines, "Cubic Spline")])
axs[0].set_title("Interpolacion equiespaciada")

# Ejercicio 2.1 inciso no equispaciado
x_points2 = chebyshev_nodes(-3, 3, 11)
y_points2 = my_function(x_points2)

x2 = np.array(x_points2)
y2 = np.array(y_points2)

poly_lagrange2 = lagrange(x2, y2)
poly_spline2 = CubicSpline(x2, y2)

y_interp_lagrange2 = poly_lagrange2(x_interval)
y_interp_splines2 = poly_spline2(x_interval)

create_line_graph(axs[1], x2, y2, [(x_interval, y_original, "func original"), (x_interval, y_interp_lagrange2, "Lagrange"),   (x_interval, y_interp_splines2, "Cubic Spline")])
axs[1].set_title("Interpolacion no equiespaciada")

# Calculamos los errores para este caso de 10 nodos
error_lagrange1 = np.abs(y_original - y_interp_lagrange)
error_lagrange2 = np.abs(y_original - y_interp_lagrange2)
error_spline1 = np.abs(y_original - y_interp_splines)
error_spline2 = np.abs(y_original - y_interp_splines2)

# Ploteamos los errores
create_error_graph(axs[2], [(x_interval, error_lagrange1, "Lagrange equispaciado error"), (x_interval, error_lagrange2, "Lagrange no-equispaciado error"), (x_interval, error_spline1, "Spline equispaciado error"), (x_interval, error_spline2, "Spline no-equispaciado error")])
axs[2].set_title("Error de polinomios")

plt.tight_layout()
plt.show()

# Calculamos errores cambiando la cantidad de nodos
def calculate_error_graph_nodes(a, b, node_function, title):
    fig, axs = plt.subplots()
    nodes_error_lagrange = {}
    nodes_error_splines = {}

    for i in range(a, b + 1):
        x_points = node_function(-3, 3, i)
        y_points = my_function(x_points)
        x_interval = np.arange(-3, 3.05, 0.05)
        poly_lagrange = lagrange(x_points, y_points)
        poly_spline = CubicSpline(x_points, y_points)

        y_interp_lagrange = poly_lagrange(x_interval)
        y_interp_splines = poly_spline(x_interval)

        nodes_error_lagrange[i] = calculate_mass_error(y_original, y_interp_lagrange)
        nodes_error_splines[i] = calculate_mass_error(y_original, y_interp_splines)

    axs.plot(list(nodes_error_lagrange.keys()), nodes_error_lagrange.values(), "-", label = "Error Lagrange")
    axs.plot(list(nodes_error_splines.keys()), nodes_error_splines.values(), "-", label = "Error Splines")
    axs.set_xlabel('Nodes')
    axs.set_ylabel('Mass error')
    axs.set_title(title)
    axs.legend()

    return axs

error_graph_x_space = calculate_error_graph_nodes(5, 13, np.linspace, "Mass error X spaced Nodes")
error_graph_chebyshev = calculate_error_graph_nodes(5, 13, chebyshev_nodes, "Mass error Chebyshev nodes")

plt.show()
