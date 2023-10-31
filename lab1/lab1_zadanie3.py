import numpy as np
import matplotlib.pyplot as plt

def h(x):
    return np.cos(x) - x

#def h(x):
    #return x**3-x**2-1



# Metoda Newtona
def newtons_method(h, h_step, x_0, tol=1e-15, max = 500):

    iterations = 0

    while abs(h(x_0)) >= tol and iterations < max:

        dh = (h(x_0)-h(x_0-h_step))/(h_step)

        x_0 = x_0 - h(x_0)/dh

        iterations += 1

    return x_0, iterations
    

exact_value = 0.739085133215
#exact_value = 1.465571231876768026656731

h_min = 1e-3
h_max = 1
h_steps = np.arange(h_min, h_max, h_min)



for x_0 in [1, 1000]:
    errors = []
    iter_counts = []
    for s in h_steps:
        root, iterations = newtons_method(h, s, x_0)
        error = abs(exact_value - root)
        errors.append(error)
        iter_counts.append(iterations)
    
    
    plt.plot(h_steps, iter_counts, label=f'x_0 = {x_0}')

plt.title("Wykres tempa zbieżności")
plt.ylabel("Liczba iteracji")
plt.xlabel("$h$")
plt.legend()

plt.show()
