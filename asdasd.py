# Update the plotting section of the code to ensure y-axis does not go below 0

# Adapted code
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np

def read_csv_to_arrays(filename, delimiter=' '):
    list1, list2 = [], []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        for row in csvreader:
            x1 = float(row[0])
            x2 = float(row[1])
            list1.append(x1)
            list2.append(x2)
    return np.array(list1), np.array(list2)

def cubic_spline_interpolation(x_data, y_data, interval=0.1, end_value=9.1):
    cs = CubicSpline(x_data, y_data)
    t_interval = np.arange(0, end_value, interval)
    return t_interval, cs(t_interval)

# Read data from CSV files
x1_data, x2_data = read_csv_to_arrays('mnyo_mediciones.csv')
x1_groundtruth, x2_groundtruth = read_csv_to_arrays('mnyo_ground_truth.csv')

# Perform cubic spline interpolation
t_interval, x1_interp_data = cubic_spline_interpolation(np.arange(len(x1_data)), x1_data)
_, x2_interp_data = cubic_spline_interpolation(np.arange(len(x2_data)), x2_data)

# Define additional lines
def vertical_line(x1_value, y_min, y_max):
    return [x1_value, x1_value], [y_min, y_max]

def diagonal_line(x1_values):
    return x1_values, 3.6 - 0.35 * x1_values

x1_min = min(min(x1_data), min(x1_groundtruth), min(x1_interp_data))
x1_max = max(max(x1_data), max(x1_groundtruth), max(x1_interp_data))
y_min = 0  # Ensure y_min is set to 0
y_max = max(max(x2_data), max(x2_groundtruth), max(x2_interp_data))

x1_vertical, x2_vertical = vertical_line(10, y_min, y_max)
x1_diagonal, x2_diagonal = diagonal_line(np.linspace(x1_min, x1_max, 500))

# Filter out the points where y is negative for the diagonal line
x1_diagonal = x1_diagonal[x2_diagonal >= 0]
x2_diagonal = x2_diagonal[x2_diagonal >= 0]

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(x1_data, x2_data, 'o', label='Data Points')
plt.plot(x1_groundtruth, x2_groundtruth, "-", label='Ground Truth')
plt.plot(x1_interp_data, x2_interp_data, "-", label='Interpolated Function')
plt.plot(x1_vertical, x2_vertical, '--', label='x1 = 10')
plt.plot(x1_diagonal, x2_diagonal, '--', label='0.35 * x1 + x2 = 3.6')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(bottom=0)  # Set the bottom limit of y-axis to 0
plt.legend()
plt.grid(True)
plt.show()
