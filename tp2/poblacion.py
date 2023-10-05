import matplotlib.pyplot as plt

def crecimiento_dif(t, N, r , K, A):
    return r * N * (1 - (N/K)) * ((N/A) - 1)

#Parametros iniciales de constantes
r = 0.1
K = 5000
A = 100

def euler_method(a, b, y0, N):
    h = (b - a) / N
    t = a
    w = y0
    values = [(t, w)]
    for i in range(N):
        w = w + h * crecimiento_dif(t, w, r, K, A)
        t = a + i * h
        values.append((t, w))
    
    return values

#Parametros iniciales de Euler
a = 0
b = 25
y0 = 500
n = 100

values = euler_method(a, b, y0, n)

# Plot the results
time_values, population_values = zip(*values)  # Unzip the values
plt.plot(time_values, population_values, label='Euler Method')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.title('Population Growth Over Time (Euler Method)')
plt.grid(True)
plt.show()
