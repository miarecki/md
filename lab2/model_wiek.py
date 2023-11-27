import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parameters
r, k, h, t  = (0.01, 10, 0.5, 200)

# Decay function
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

# Euler method for implicit decay using trapezium method
def euler_trapezium(y, a, b, h):
    for i in range(a, b):
        g = lambda x: x - y[i - 1] - h * (decay_fun(x, y[i - a + 1]) + decay_fun(y[i - 1], y[i - a])) / 2
        y[i] = fsolve(g, y[i - 1])[0]
    return y


# Initialize the subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

cnt_1 = 0
for tau in [1.5, 11.5, 17]:
    cnt_2 = 0
    j=np.random.random(1)
    for i in range(3):
        # Initialize result vectors
        y_left = np.arange(0, t + h, h)
        y_right = np.arange(0, t + h, h)
        y_trapezium = np.arange(0, t + h, h)

        # Initial conditions
        init_random = i * (np.sort(np.random.random(int(tau / h)))) 
        # dla i = 0 by nie bylo stale rowne 0 to + j
        y_left[:int(tau / h)] = init_random + j
        y_right[:int(tau / h)] = init_random + j
        y_trapezium[:int(tau / h)] = init_random + j

        # Compute the solutions after tau
        y_left = euler_left(y_left, int(tau / h), int(t / h) + 1, h)
        y_right = euler_right(y_right, int(tau / h), int(t / h) + 1, h)
        y_trapezium = euler_trapezium(y_trapezium, int(tau / h), int(t / h) + 1, h)

        # Plot the results against time
        axs[cnt_1, cnt_2].plot(np.arange(0, t + h, h), y_left, label='left')
        axs[cnt_1, cnt_2].plot(np.arange(0, t + h, h), y_right, label='right')
        axs[cnt_1, cnt_2].plot(np.arange(0, t + h, h), y_trapezium, label='trapezium')

        axs[cnt_1, cnt_2].set_title(f'$\\tau = {tau}$')

        cnt_2 += 1

    cnt_1 += 1

plt.legend()
plt.show()


print(j)