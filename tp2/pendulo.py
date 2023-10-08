import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Resolver analiticamente pendulo linealizado
def model(y, t, g, l):
    theta_prime, theta_double_prime = y  # theta_prime represents u and theta_double_prime represents v
    du_dt = theta_double_prime
    dv_dt = -(g / l) * theta_prime
    return [du_dt, dv_dt]

# Parametros
g = 9.81  # Acceleration due to gravity (m/s^2)
l = 1.0   # Length of the pendulum (m)
theta0 = 0.1  # Initial angular displacement (45 degrees)
theta_prime0 = 0.0  # Initial angular velocity (rad/s)

# Tiempo
t = np.linspace(0, 20, 1001)

# Condiciones Iniciales
initial_conditions = [theta0, theta_prime0]

# Resolviendo con odeint
solution = odeint(model, initial_conditions, t, args=(g, l))

# Resultados
theta_prime_values, theta_double_prime_values = solution[:, 0], solution[:, 1]

# Plot the angular displacement over time
plt.plot(t, theta_prime_values)
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (radians)')
plt.title('Simple Pendulum Motion (Vector Form)')
plt.show()

