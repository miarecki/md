import numpy as np
import matplotlib.pyplot as plt

# Funkcja obliczająca y = 1/√x bezpośrednio
def calculate_direct(x):
    return 1/np.sqrt(x)

# metoda Newtona
def calculate_newton(x, n):
    y = 0.01 #dobry guess
    for _ in range(n):
        y = 0.5 * y * (3 - x*y**2)
    return y


x_values = np.linspace(1, 1000, 100)


iterations = [1, 2, 3, 4]

#  wykresy
for n in iterations:
    direct_results = [calculate_direct(x) for x in x_values]
    newton_results = [calculate_newton(x, n) for x in x_values]
    errors = [abs(direct - newton) for direct, newton in zip(direct_results, newton_results)]
    
    plt.plot(x_values, errors, label=f'{n} iteracji')

plt.xlabel('Wartość x')
plt.ylabel('Błąd bezwzględny')
plt.legend()
plt.show()

