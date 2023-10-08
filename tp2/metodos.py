import numpy as np
import matplotlib.pyplot as plt

def pendulo(y, g, l):
    return -(g / l) * np.sin(y)

def euler_method(a, b, y0, y_diff0, N):
    h = (b - a) / N
    t = a
    w = y0
    w_diff = y_diff0
    values = [(t, w, w_diff)]
    for _ in range(N):
        new_w_diff = w_diff + h * (pendulo(w, 9.81, 1.0))
        w = w + h * w_diff

        w_diff = new_w_diff
        t += h
        values.append((t, w, new_w_diff))
    
    return values

# Define the initial conditions
a = 0.0  # Initial time
b = 15.0  # Final time
y0 = 0.1  # Initial angular displacement (45 degrees)
y_diff0 = 0.0  # Initial angular velocity (rad/s)
N = 1000  # Number of time steps

# Use Euler's method to solve the equation
result = euler_method(a, b, y0, y_diff0, N)

# Extract time and angular displacement data for plotting
time_values, angular_displacement_values, angular_velocity_values = zip(*result)

# Parameters
m = 1.0  # Mass of the pendulum bob
l = 1.0  # Length of the pendulum (m)
g = 9.81  # Acceleration due to gravity (m/s^2)

# Calculate total energy E at each time step
T_values = 0.5 * m * l**2 * np.array(angular_velocity_values)**2
V_values = -m * g * l * np.cos(np.array(angular_displacement_values))
E_values = T_values + V_values

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Pendulum Motion Plot (Full Row)
axs[0, 0].plot(time_values, angular_displacement_values)
axs[0, 0].set_ylabel('Angular Displacement (radians)')
axs[0, 0].set_title('Pendulum Motion')

# Total Energy (E) Plot (Full Row)
axs[1, 0].plot(time_values, E_values, color='blue')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Total Energy (Joules)')
axs[1, 0].set_title('Total Energy of Simple Pendulum')

# Kinetic Energy (T) Plot (Shared)
axs[0, 1].plot(time_values, T_values, color='green')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Kinetic Energy (Joules)')

# Potential Energy (V) Plot (Shared)
axs[1, 1].plot(time_values, V_values, color='red')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Potential Energy (Joules)')
plt.tight_layout()
plt.show()

#Runge Kutta

def function_vec(θ, θ_dot):
    return np.array([pendulo(θ, g, l), θ_dot])

def rungekutta(θ, h):
    k1 = h * function_vec(θ)
    k2 = h * function_vec(θ + 0.5 * k1)
    k3 = h * function_vec(θ + 0.5 * k2)
    k4 = h * function_vec(θ + k3)

    return θ + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Use Runge-Kutta method to solve the equation
result_rk = []
theta = np.array([y0, y_diff0])  # Initial state vector [angular displacement, angular velocity]
h = (b - a) / N  # Step size

for _ in range(N):
    result_rk.append((a, theta[0], theta[1]))
    theta = rungekutta(theta[0], theta[1], h)
    a += h

# Extract time and angular displacement data for plotting
time_values_rk, angular_displacement_values_rk, angular_velocity_values_rk = zip(*result_rk)

# Calculate kinetic energy (T) using NumPy operations
T_values_rk = 0.5 * m * l**2 * np.array(angular_velocity_values_rk)**2

# Calculate potential energy (V) using NumPy operations
V_values_rk = -m * g * l * np.cos(np.array(angular_displacement_values_rk))

# Calculate total energy (E) as the sum of kinetic and potential energy
E_values_rk = T_values_rk + V_values_rk

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Pendulum Motion Plot (Full Row)
axs[0, 0].plot(time_values_rk, angular_displacement_values_rk)
axs[0, 0].set_ylabel('Angular Displacement (radians)')
axs[0, 0].set_title('Pendulum Motion (Runge-Kutta)')

# Total Energy (E) Plot (Full Row)
axs[1, 0].plot(time_values_rk, E_values_rk, color='blue')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Total Energy (Joules)')
axs[1, 0].set_title('Total Energy of Simple Pendulum (Runge-Kutta)')

# Kinetic Energy (T) Plot (Shared)
axs[0, 1].plot(time_values_rk, T_values_rk, color='green')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Kinetic Energy (Joules)')
axs[0, 1].set_title('Kinetic Energy (Runge-Kutta)')

# Potential Energy (V) Plot (Shared)
axs[1, 1].plot(time_values_rk, V_values_rk, color='red')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Potential Energy (Joules)')
axs[1, 1].set_title('Potential Energy (Runge-Kutta)')

plt.tight_layout()
plt.show()



