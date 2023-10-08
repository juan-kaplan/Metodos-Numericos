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


def runge_kutta_4(a, b, y0, N):
    h = (b - a) / N 
    t = a  
    w = y0  
    values = [(t, w)]  
    
    for i in range(N):
        k1 = h * crecimiento_dif(t, w, r, K, A)
        k2 = h * crecimiento_dif(t + 0.5*h, w + 0.5*k1, r, K, A)
        k3 = h * crecimiento_dif(t + 0.5*h, w + 0.5*k2, r, K, A)
        k4 = h * crecimiento_dif(t + h, w + k3, r, K, A)
        
        w = w + (k1 + 2*k2 + 2*k3 + k4) / 6  
        t = a + i * h  
        values.append((t, w))  
    
    return values


# Parametros para testear ambos metodos y comparar
a = 0
b = 25
y0 = 500
steps = [10, 50, 100, 500]

# Hace todas las cuentas y plottea euler Y runge con distintos step sizes
for n in steps:
    euler_values = euler_method(a, b, y0, n)
    rk4_values = runge_kutta_4(a, b, y0, n)
    
    # Plottea los resultados
    time_values_euler, population_values_euler = zip(*euler_values) #Estaria bueno mencionar que pasa con euler cuando son 50 steps
    time_values_rk4, population_values_rk4 = zip(*rk4_values)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_values_euler, population_values_euler, label='Euler Method, N={}'.format(n))
    plt.plot(time_values_rk4, population_values_rk4, label='RK4 Method, N={}'.format(n))
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.legend()
    plt.title('Population Growth Over Time (Euler vs RK4), N={}'.format(n))
    plt.grid(True)
    plt.show()

#DESTACO: cuanto mayor es la cantidad de steps, mas consistente es la aprox. Euler y Runge parecen estar muy de acuerdo en 500 steps, lo
#cual me hace pensar que es un buen parametro para la siguiente parte, antes de 500 es medio cualquier cosa. testear con distintos steps sirve para despues saber 
#cual usar como parametro en la parte que viene:


# Vary initial population size
initial_populations = [-10, 99, 100, 110, 300, 400, 500, 1000, 5000, 7000] #Exagere con la cantidad de pruebas, pero hay algunos importantes a destacar: 99, 100, 1000, 5000, 7000. 

# Iterate over different initial population sizes
for y0 in initial_populations:
    # Get solutions using both methods
#    euler_values = euler_method(a, b, y0, 500)
    rk4_values = runge_kutta_4(a, b, y0, 500) #Solamente plotteo con runge porque son casi identicos ocn el euler. Ademas, euler se cayo a pedazos con N bajo, asi que me hace creer que es un poquito menos preciso que runge aun con 500 steps.
    
    # Extract and plot the numerical solutions
#    time_values_euler, population_values_euler = zip(*euler_values)
    time_values_rk4, population_values_rk4 = zip(*rk4_values)
    
    plt.figure(figsize=(12, 6))
#    plt.plot(time_values_euler, population_values_euler, label='Euler, Initial population={}'.format(y0))
    plt.plot(time_values_rk4, population_values_rk4, label='RK4, Initial population={}'.format(y0))
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.legend()
    plt.title('Population Growth Over Time with Different Initial Populations')
    plt.grid(True)
    plt.show()

