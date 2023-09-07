import csv
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fixed_point
import numpy as np

mediciones_file = "mnyo_mediciones.csv"
ground_truth = "mnyo_ground_truth.csv"

def create_error_graph(axs, functions, title=""):
    for function in functions:
        axs.plot(function[0], function[1], '-', label= function[2])

    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()
    axs.grid(True)
    return axs

def create_line_graph(axs, x_datapoints, y_datapoints, functions, title=""):
    axs.plot(x_datapoints, y_datapoints, "o", label="Data Points")
    for function in functions:
        axs.plot(function[0], function[1], "-", label=function[2])

    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    axs.legend()
    return axs


x1_data = []
x2_data = []

x1_groundtruth = []
x2_groundtruth = []


def csv_reader(filename, list1, list2):
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ")

        for row in csvreader:
            x1 = float(row[0])
            x2 = float(row[1])

            list1.append(x1)
            list2.append(x2)


csv_reader(mediciones_file, x1_data, x2_data)
csv_reader(ground_truth, x1_groundtruth, x2_groundtruth)

time_values = list(range(10))
x1_interpolated = CubicSpline(time_values, x1_data)
x2_interpolated = CubicSpline(time_values, x2_data)

t_interval = np.arange(0, 9.1, 0.1)
x1_interp_data = x1_interpolated(t_interval)
x2_interp_data = x2_interpolated(t_interval)

fig, axs = plt.subplots(figsize=(10, 8))

plt.plot(x1_data, x2_data, "o", label="Data Points")
plt.plot(x1_groundtruth, x2_groundtruth, "-", label="Ground Truth")
plt.plot(x1_interp_data, x2_interp_data, "-", label="Interpolated function")

axs.axvline(x=10, color="red", linestyle="-", label="x1 = 10")
x1_line = np.linspace(min(x1_data), max(x1_data), 100)
x2_line = 3.6 - 0.35 * x1_line
axs.plot(x1_line, x2_line, "-", label="0.35 * x1 + x2 = 3.6")

# Labels and legend
plt.xlabel("X1")
plt.ylabel("X2")
axs.set_ylim(bottom=0)
plt.legend()
plt.grid(True)

# Busqueda de intersecciones
def x10_intersection_poly(t):
    return (x1_interpolated(t) - 10)

def linear_intersection(t):
    return x2_interpolated(t) + 0.35 * x1_interpolated(t) - 3.6

def linear_intersection_deriv(t):
    return 0.35 * x1_interpolated.derivative()(t) + x2_interpolated.derivative()(t)

def newtonRaphson(func, derivFunc, x, epsilon):
    iterations = [x]
    h = func(x) / derivFunc(x)
    while abs(h) >= epsilon:
        h = func(x)/derivFunc(x)
        pre_x = x
        x = x - h
        iterations += [abs(pre_x - x)]
    
    return x, iterations

t_intersection_value1, iterations_1 = newtonRaphson(x10_intersection_poly, x1_interpolated.derivative(), 3, 0.001)
t_intersection_value2, iterations_2 = newtonRaphson(linear_intersection, linear_intersection_deriv, 0, 0.001)
t_intersection_value3, iterations_3 = newtonRaphson(linear_intersection, linear_intersection_deriv, 1, 0.001)
t_intersection_value4, iterations_4 = newtonRaphson(linear_intersection, linear_intersection_deriv, 2, 0.001)

print(t_intersection_value2)
x1_intersections = [x1_interpolated(t_intersection_value2), x1_interpolated(t_intersection_value3), x1_interpolated(t_intersection_value4),x1_interpolated(t_intersection_value1)]
x2_intersections = [x2_interpolated(t_intersection_value2), x2_interpolated(t_intersection_value3), x2_interpolated(t_intersection_value4),x2_interpolated(t_intersection_value1),]

list1 = [t_intersection_value1, t_intersection_value2, t_intersection_value3, t_intersection_value4]
plt.plot(x1_intersections, x2_intersections, "o", label = "Intersection")

plt.legend()
plt.show()

# Create a subplot for the Newton-Raphson convergence plot
fig, axs_convergence = plt.subplots(figsize=(10, 6))
plt.title("Newton-Raphson Convergence")

# Plot convergence for iteration 1
axs_convergence.plot(range(len(iterations_1)), iterations_1, label="Iteration 1", marker='o')

# Plot convergence for iteration 2
axs_convergence.plot(range(len(iterations_2)), iterations_2, label="Iteration 2", marker='o')

# Plot convergence for iteration 3
axs_convergence.plot(range(len(iterations_3)), iterations_3, label="Iteration 3", marker='o')

# Plot convergence for iteration 4
axs_convergence.plot(range(len(iterations_4)), iterations_4, label="Iteration 4", marker='o')

axs_convergence.set(xlabel="Iteration", ylabel="Pn - Pn-1")

# Add legend
plt.legend()

# Show the convergence plot
plt.show()

# Encontrar los verdaderos puntos de interseccion
# Interpolacion de la ground truth para aproximar valores de por medio

time_values = list(np.linspace(0, 9, 100))
print(len(time_values))
x1_truth = CubicSpline(time_values, x1_groundtruth)
x2_truth = CubicSpline(time_values, x2_groundtruth)

def x10_truth_poly(t):
    return (x1_truth(t) - 10)

def linear_truth_intersection(t):
    return x2_truth(t) + 0.35 * x1_truth(t) - 3.6

def linear_truth_intersection_deriv(t):
    return 0.35 * x1_truth.derivative()(t) + x2_truth.derivative()(t)

t_intersection_value5, iterations_5 = newtonRaphson(x10_truth_poly, x1_truth.derivative(), 3, 0.001)
t_intersection_value6, iterations_6 = newtonRaphson(linear_truth_intersection, linear_truth_intersection_deriv, 0, 0.001)
t_intersection_value7, iterations_7 = newtonRaphson(linear_truth_intersection, linear_truth_intersection_deriv, 1.5, 0.001)
t_intersection_value8, iterations_8 = newtonRaphson(linear_truth_intersection, linear_truth_intersection_deriv, 2.5, 0.001)

print(t_intersection_value6, t_intersection_value7, t_intersection_value8, t_intersection_value5)
x1_intersections2 = [x1_truth(t_intersection_value6), x1_truth(t_intersection_value7), x1_truth(t_intersection_value8),x1_truth(t_intersection_value5)]
x2_intersections2 = [x2_truth(t_intersection_value6), x2_truth(t_intersection_value7), x2_truth(t_intersection_value8),x2_truth(t_intersection_value5),]
print(x1_intersections)
print(x1_intersections2)
print(x2_intersections)
print(x2_intersections2)

#Ver comparacion de error entre las dos
x1_original_data = x1_truth(t_interval)
x2_original_data = x2_truth(t_interval)
error_x1 = np.abs(x1_original_data - x1_interp_data)
error_x2 = np.abs(x2_original_data - x2_interp_data)

fig, axs_error = plt.subplots(figsize=(10, 6))
create_error_graph(axs_error, [(t_interval, error_x1, "x1_error"), (t_interval, error_x2, "x2_error")])
plt.show()