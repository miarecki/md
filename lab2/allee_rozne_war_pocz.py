import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

def Euler(f, y_0, a, b, h, method):
    num_steps = int((b - a) / h) + 1
    
    t_values = np.linspace(a, b, num_steps)
    y_values = np.zeros(num_steps)
    
    y_values[0] = y_0
    
    # lewostronna jawna
    if method == 'left':
        for i in range(1, num_steps):
            y_values[i] = y_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1])
    
    # prawo niejawna
    if method == 'right':
        for i in range(1, num_steps):
            implicit_function = lambda y_next: y_values[i - 1] + h * f(t_values[i], y_next) - y_next
            y_values[i] = fsolve(implicit_function, y_values[i - 1])[0]
    #trapez
    if method == 'trapezium':
        for i in range(1, num_steps):
            implicit_function = lambda y_next: y_values[i - 1] + h * (f(t_values[i], y_next) + f(t_values[i - 1], y_values[i - 1]))/2 - y_next
            y_values[i] = fsolve(implicit_function, y_values[i - 1])[0]

    return y_values


# RHS
def f(t, y):
    return y*1*(1-y)*(2*y - 1)

h = 0.1
K = 1
A = 0.5



#NEED
'''
for method in ["left", "right", "trapezium"]:
    for y_0 in np.arange(-.5, 1.5, h):
        plt.plot(np.arange(0, 8+h, h), Euler(f, y_0, 0, 8, h, method=method))
    plt.xlabel('t')
    plt.ylabel('N(t)')
    plt.xlim(0,8)
    plt.grid(True)
    plt.show()
'''
# 3/2 --> lambda t: (2*np.sqrt(np.e**t*(4*np.e**t-3)) + 4*np.e**t - 3)/(8*np.e**t - 6) 
# 4/3 --> lambda t: (5*np.sqrt(np.e**t*(25*np.e**t-16)) + 25*np.e**t - 16)/(50*np.e**t - 32)]
# lambda t: (6*np.sqrt(np.e**t*(36*np.e**t-11)) + 36*np.e**t - 11)/(72*np.e**t - 22)] #11/10


#NEED

y_dokladne = [lambda t: (-0.5*np.e**(.5*t)*np.sqrt(np.sqrt(3) + np.e**t) + 0.5*np.e**t + 8/9)/(np.sqrt(3) + np.e**t),
  lambda t: (3*np.sqrt(np.e**t*(9*np.e**t+16)) + 9*np.e**t + 16)/(18*np.e**t + 32),
  lambda t: (5*np.sqrt(np.e**t*(25*np.e**t-16)) + 25*np.e**t - 16)/(50*np.e**t - 32)]

l = 0
for y_0 in [1/5, 4/5, 11/10]:
    for method in ["left", "right", "trapezium"]:
        error = []
        for h in np.arange(10**(-1), 1.0, 10**(-3)):
            t_values, y_approx_values = np.linspace(0, 10, int(10/h) + 1), Euler(f, y_0, 0, 10, h, method=method)
            y_exact_values = y_dokladne[l](t_values)
            
            error.append(np.mean(np.abs(y_approx_values - y_exact_values)))
        plt.plot(np.arange(10**(-1), 1.0, 10**(-3)), error, label=method)

    l += 1
    plt.xlabel("$h$")
    plt.ylabel("Błąd względny")
    plt.legend()
    plt.show()




'''takes a min
h = 0.0001
times = []
for method in ["left", "right", "trapezium"]:
	for y_0 in [1/5, 4/5, 4/3]:
		start = time.time()
		Euler(f, y_0, 0, 10, h, method=method)
		end = time.time()
		times.append([method, y_0, round(end - start, 5)])
print(times)
'''


'''
data = [['left', 0.2, 0.07855], ['left', 0.8, 0.06303], ['left', 1.33, 0.07817],
        ['right', 0.2, 6.34245], ['right', 0.8, 6.13749], ['right', 1.33, 6.16931],
        ['trapezium', 0.2, 8.39975], ['trapezium', 0.8, 7.99121], ['trapezium', 1.33, 8.13473]]


methods = [entry[0] for entry in data]
y_values = [entry[1] for entry in data]
execution_times = [entry[2] for entry in data]


fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(data)), execution_times, color=['blue', 'green', 'red']*3)
ax.set_xticks(range(len(data)))
ax.set_xticklabels([f"{method}\n{y}" for method, y, _ in data])
ax.set_ylabel('Czas wykonania (s)')
#ax.set_title('Execution Time for Different Methods and Initial Values')

# Display the plot
plt.show()
'''
