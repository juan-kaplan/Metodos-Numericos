import csv
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np

def create_line_graph(axs, x_datapoints, y_datapoints, functions, title = ""):
    axs.plot(x_datapoints, y_datapoints, 'o', label='Data Points')
    for function in functions:
        axs.plot(function[0], function[1], '-', label= function[2])
  
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.legend()
    return axs

mediciones_file = 'mnyo_mediciones.csv'
ground_truth = "mnyo_ground_truth.csv"

x1_data = []
x2_data = []

x1_groundtruth = []
x2_groundtruth = []

def csv_reader(filename, list1, list2):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        
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

plt.plot(x1_data, x2_data, 'o', label='Data Points')
plt.plot(x1_groundtruth, x2_groundtruth, "-", label = "Ground Truth")
plt.plot(x1_interp_data, x2_interp_data, "-", label= "Interpolated function")
plt.xlabel('X1')
plt.ylabel('X2')

axs.axvline(x=10, color='red', linestyle='-', label='x1 = 10')
x1_line = np.linspace(min(x1_data), max(x1_data), 100) 
x2_line = 3.6 - 0.35 * x1_line 
axs.plot(x1_line, x2_line, "-", label='0.35 * x1 + x2 = 3.6')

# Labels and legend
plt.xlabel('X1')
plt.ylabel('X2')
axs.set_ylim(bottom=0)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()



