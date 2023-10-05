import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from numpy.polynomial.polynomial import Polynomial

# Definicion de funcion f2
def f2(x1, x2):
    term1 = 0.7 * np.exp(-((9*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)
    term2 = 0.45 * np.exp(-((9*x1 + 1)**2)/9 - ((9*x2 + 1)**2)/5)
    term3 = 0.55 * np.exp(-((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4)
    term4 = -0.01 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    return term1 + term2 + term3 + term4

# Espacio equisespaciado
n = 9
x1_vals = np.linspace(-1, 1, n+1)
x2_vals = np.linspace(-1, 1, n+1)

# Calculo de funcion
f_values = np.array([[f2(x1, x2) for x1 in x1_vals] for x2 in x2_vals])

# Compuatando polinomios de lagrange
basis_x1 = [Polynomial.fromroots(np.delete(x1_vals, i)) for i in range(n+1)]
basis_x2 = [Polynomial.fromroots(np.delete(x2_vals, i)) for i in range(n+1)]

# Normalizando las bases
for i in range(n+1):
    basis_x1[i] /= basis_x1[i](x1_vals[i])
    basis_x2[i] /= basis_x2[i](x2_vals[i])

# Evaluamos interpolacion lagrange
x1_fine = np.linspace(-1, 1, 100)
x2_fine = np.linspace(-1, 1, 100)
X_fine, Y_fine = np.meshgrid(x1_fine, x2_fine)

Z_fine = np.zeros_like(X_fine)

for i in range(n+1):
    for j in range(n+1):
        Z_fine += f_values[j, i] * np.outer(basis_x2[j](x2_fine), basis_x1[i](x1_fine))

# Cubic Splines
spline = RectBivariateSpline(x1_vals, x2_vals, f_values)

Z_fine_spline = spline(x1_fine, x2_fine)

# Graficos
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

axs[0].plot_surface(X_fine, Y_fine, np.array([[f2(x1, x2) for x1 in x1_fine] for x2 in x2_fine]), cmap='viridis')
axs[0].set_xlabel('x1')
axs[0].set_ylabel('x2')
axs[0].set_zlabel('f2(x1, x2)')
axs[0].set_title('Original Function f2(x1, x2)')

axs[1].plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis')
axs[1].set_xlabel('x1')
axs[1].set_ylabel('x2')
axs[1].set_zlabel('f2(x1, x2)')
axs[1].set_title('2D Lagrange Interpolated Function with Evenly Spaced Nodes')


axs[2].plot_surface(X_fine, Y_fine, Z_fine_spline, cmap='viridis')
axs[2].set_xlabel('x1')
axs[2].set_ylabel('x2')
axs[2].set_zlabel('f2(x1, x2)')
axs[2].set_title('Cubic Spline Interpolated Function with Evenly Spaced Nodes')

#Error maximo
f2_fine = np.array([[f2(x1, x2) for x1 in x1_fine] for x2 in x2_fine])

print("Lagrange:", np.max(np.abs(f2_fine - Z_fine)), "Splines:", np.max(np.abs(f2_fine - Z_fine_spline)))


plt.tight_layout()
plt.show()

# Computar Mass error
def calculate_mass_error(fine, interp):
    return np.sum(np.abs(fine - interp))

mass_errors_lagrange = {}
mass_errors_spline = {}

for n in range(5, 21):  # 5x5, 6x6, ..., 20x20
    x1_vals = np.linspace(-1, 1, n)
    x2_vals = np.linspace(-1, 1, n)
    
    f_values = np.array([[f2(x1, x2) for x1 in x1_vals] for x2 in x2_vals])
    
    # Lagrange interpolacion
    basis_x1 = [Polynomial.fromroots(np.delete(x1_vals, i)) for i in range(n)]
    basis_x2 = [Polynomial.fromroots(np.delete(x2_vals, i)) for i in range(n)]
    
    for i in range(n):
        basis_x1[i] /= basis_x1[i](x1_vals[i])
        basis_x2[i] /= basis_x2[i](x2_vals[i])
    
    x1_fine = np.linspace(-1, 1, 100)
    x2_fine = np.linspace(-1, 1, 100)
    X_fine, Y_fine = np.meshgrid(x1_fine, x2_fine)
    
    Z_fine = np.zeros_like(X_fine)
    
    for i in range(n):
        for j in range(n):
            Z_fine += f_values[j, i] * np.outer(basis_x2[j](x2_fine), basis_x1[i](x1_fine))
            
    # Cubic Spline interpolacion
    spline = RectBivariateSpline(x1_vals, x2_vals, f_values)
    Z_fine_spline = spline(x1_fine, x2_fine)
    
    # Funcion original
    f2_fine = np.array([[f2(x1, x2) for x1 in x1_fine] for x2 in x2_fine])
    
    # Calcular mass error
    mass_errors_lagrange[n] = calculate_mass_error(f2_fine, Z_fine)
    mass_errors_spline[n] = calculate_mass_error(f2_fine, Z_fine_spline)

# Plotting
fig, axs = plt.subplots()
axs.plot(list(mass_errors_lagrange.keys()), list(mass_errors_lagrange.values()), label='Lagrange')
axs.plot(list(mass_errors_spline.keys()), list(mass_errors_spline.values()), label='Cubic Spline')
axs.set_xlabel('Node Count (NxN)')
axs.set_ylabel('Mass Error')
axs.set_title('Mass Error vs Node Count')
axs.legend()
axs.set_yscale('log')

plt.show()