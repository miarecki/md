import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

r, k, h, t  = (0.01, 10, 0.5, 200)

def decay_fun(x, y):
    return r * x * (k - y)

# left
def euler_left(y, a, b, h):
    for i in range(a, b):
        y[i] = y[i - 1] + h * decay_fun(y[i - 1], y[i - a])
    return y

# right
def euler_right(y, a, b, h):
    for i in range(a, b):
        g = lambda x: x - y[i - 1] - h * decay_fun(x, y[i - a + 1])
        y[i] = fsolve(g, y[i - 1])[0]
    return y

# trap
def euler_trapezium(y, a, b, h):
    for i in range(a, b):
        g = lambda x: x - y[i - 1] - h * (decay_fun(x, y[i - a + 1]) + decay_fun(y[i - 1], y[i - a])) / 2
        y[i] = fsolve(g, y[i - 1])[0]
    return y



fig, axs = plt.subplots(3, 3, figsize=(15, 15))

wiersz = 0
for tau in [1.5, 11.5, 17]:
    kolumna = 0
    j=np.random.random(1)
    for i in range(3):
        y_left = np.arange(0, t + h, h)
        y_right = np.arange(0, t + h, h)
        y_trapezium = np.arange(0, t + h, h)
        init_random = i * (np.sort(np.random.random(int(tau / h)))) 
        # dla i = 0 by nie bylo stale rowne 0 to + j
        y_left[:int(tau / h)] = init_random + j
        y_right[:int(tau / h)] = init_random + j
        y_trapezium[:int(tau / h)] = init_random + j
        y_left = euler_left(y_left, int(tau / h), int(t / h) + 1, h)
        y_right = euler_right(y_right, int(tau / h), int(t / h) + 1, h)
        y_trapezium = euler_trapezium(y_trapezium, int(tau / h), int(t / h) + 1, h)
        axs[wiersz, kolumna].plot(np.arange(0, t + h, h), y_left, label='left')
        axs[wiersz, kolumna].plot(np.arange(0, t + h, h), y_right, label='right')
        axs[wiersz, kolumna].plot(np.arange(0, t + h, h), y_trapezium, label='trapezium')
        axs[wiersz, kolumna].set_title(f'$\\tau = {tau}$')

        kolumna += 1

    wiersz += 1

plt.legend()
plt.show()


#print(j)