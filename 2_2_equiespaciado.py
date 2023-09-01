import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

# Define the function f2(x1, x2)
def f2(x1, x2):
    term1 = 0.7 * np.exp(-((9*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)
    term2 = 0.45 * np.exp(-((9*x1 + 1)**2)/9 - ((9*x2 + 1)**2)/5)
    term3 = 0.55 * np.exp(-((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4)
    term4 = -0.01 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    return term1 + term2 + term3 + term4

# Create a meshgrid for the interpolation
x1_vals = np.linspace(-1, 1, 10)
x2_vals = np.linspace(-1, 1, 10)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)

# Calculate function values for each combination of x1 and x2
f_values = np.array([[f2(x1, x2) for x1 in x1_vals] for x2 in x2_vals])

# Perform cubic spline interpolation
spline = RectBivariateSpline(x1_vals, x2_vals, f_values)

# Perform 1D Lagrange interpolation for each row of the data grid
row_interpolated_data = np.zeros((len(x2_vals), len(x1_vals)))
for i, x2 in enumerate(x2_vals):
    poly = lagrange(x1_vals, f_values[i])
    row_interpolated_data[i] = poly(x1_vals)

# Perform 1D Lagrange interpolation for each column of the row-interpolated data
final_interpolated_data = np.zeros((len(x2_vals), len(x1_vals)))
for i, x1 in enumerate(x1_vals):
    poly = lagrange(x2_vals, row_interpolated_data[:, i])
    final_interpolated_data[:, i] = poly(x2_vals)

# Define a finer grid for the plots
x1_fine = np.linspace(-1, 1, 100)
x2_fine = np.linspace(-1, 1, 100)

# Evaluate the cubic spline interpolated function at the finer grid
Z_fine = spline(x1_fine, x2_fine)

# Perform 1D Lagrange interpolation for each row of the data grid
row_interpolated_data_fine = np.zeros((len(x2_vals), len(x1_fine)))
for i, x2 in enumerate(x2_vals):
    poly = lagrange(x1_vals, f_values[i])
    row_interpolated_data_fine[i] = poly(x1_fine)

# Perform 1D Lagrange interpolation for each column of the row-interpolated data
final_interpolated_data_fine = np.zeros((len(x2_fine), len(x1_fine)))
for i, x1 in enumerate(x1_fine):
    poly = lagrange(x2_vals, row_interpolated_data_fine[:, i])
    final_interpolated_data_fine[:, i] = poly(x2_fine)

# Create a meshgrid for the plots
X_fine, Y_fine = np.meshgrid(x1_fine, x2_fine)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})

# Plot the surface of the original function
axs[0, 0].plot_surface(X_fine, Y_fine, np.array([[f2(x1, x2) for x1 in x1_fine] for x2 in x2_fine]), cmap='viridis')
axs[0, 0].set_xlabel('x1')
axs[0, 0].set_ylabel('x2')
axs[0, 0].set_zlabel('f2(x1, x2)')
axs[0, 0].set_title('Original Function f2(x1, x2)')

# Plot the surface of the Lagrange interpolated function
axs[0, 1].plot_surface(X_fine, Y_fine, final_interpolated_data_fine, cmap='viridis')
axs[0, 1].set_xlabel('x1')
axs[0, 1].set_ylabel('x2')
axs[0, 1].set_zlabel('f2(x1, x2)')
axs[0, 1].set_title('Lagrange Interpolated Function')

# Plot the surface of the cubic spline interpolated function
axs[1, 0].plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis')
axs[1, 0].set_xlabel('x1')
axs[1, 0].set_ylabel('x2')
axs[1, 0].set_zlabel('f2(x1, x2)')
axs[1, 0].set_title('Cubic Spline Interpolated Function')

# Remove the last plot
fig.delaxes(axs[1, 1])

plt.tight_layout()
plt.show()
