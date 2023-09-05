import csv
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
import numpy as np

mediciones_file = "mnyo_mediciones.csv"
ground_truth = "mnyo_ground_truth.csv"


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
plt.xlabel("X1")
plt.ylabel("X2")

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

def first_intersection_poly(t):
    return (x1_interpolated(t) - 10)

def second_intersection_poly(t):
    x1_value = x1_interpolated(t)
    return x2_interpolated(t) + (- 0.35 * x1_value) + 3.6

def second_derivative(t):
    x1_derivative = x1_interpolated.derivative()
    x2_derivative = x2_interpolated.derivative()

    return x1_derivative(t) + 0.35 * x2_derivative(t)

def newtonRaphson(func, derivFunc, x, s): 
    for m in range(s):
        h = func(x) / derivFunc(x) 
        if(derivFunc(x) !=0 and abs(h) >= 0.0001):
            x = x - h 
        elif(abs(h) <= 0.0001):
            break;
    
    return x

t_intersection_value1 = newtonRaphson(first_intersection_poly, x1_interpolated.derivative(), 9, 100)
#t_intersection_value2 = newton(func = second_intersection_poly, fprime= second_derivative,x0= 2, tol= 0.001)

print(x1_interpolated(t_intersection_value1))

x1_intersections = [x1_interpolated(t_intersection_value1), x1_interpolated(t_intersection_value1)]
x2_intersections = [x2_interpolated(t_intersection_value1), x2_interpolated(t_intersection_value1)]
plt.plot(x1_interpolated(t_intersection_value1), x2_interpolated(t_intersection_value1), "o", label = "Intersection")

plt.plot(x1_interpolated(t_intersection_value1), x2_interpolated(t_intersection_value1), "o", label = "Intersection")

plt.show()