import numpy as np
import matplotlib.pyplot as plt

def rectangle_solver(function_values, h, method):
    if method == "rightpoint":
        return np.sum(function_values[:-1] * h)
    if method == "leftpoint":
        return np.sum(function_values[1:] * h)
    

def trapezium_solver(function_values, h):
    return (np.sum(function_values) - 0.5 * (function_values[0] + function_values[-1])) * h



f = lambda x: np.cos(x)
exact_value = 2*np.sin(1)


'''
f = lambda x: np.cos(100*x)
exact_value = np.sin(100)/50
'''

'''
f = lambda x: np.where(x == 0, 1, np.sin(x) / x)
exact_value = 1.89216
'''

error_r, error_l, error_t = [], [], []



h_values = np.arange(0.01, 1.01, 0.01)

for h in h_values:
    domain = np.linspace(-1, 1, num=int(2/h+1))
    error_r.append(np.abs(exact_value - rectangle_solver(f(domain), h, "rightpoint")))
    error_l.append(np.abs(exact_value - rectangle_solver(f(domain), h, "leftpoint")))
    error_t.append(np.abs(exact_value - trapezium_solver(f(domain), h)))


plt.plot(h_values, error_r, label="M. prostokatow prawostr.")
plt.plot(h_values, error_l, label="M. prostokatow lewostr.")
plt.plot(h_values, error_t, label="M. trapezów")
plt.title("Wykres błędu bezwzględnego")
plt.ylabel("Błąd bezwzględny")
plt.xlabel("$h$")
plt.legend()

plt.show()

