import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

#funkcje z zadania


f = lambda x: np.where(x == 0, 1, np.sin(x) / x)
exact_value = lambda x: np.where(x == 0, 0, (x*np.cos(x) - np.sin(x))/(x**2))



x_range = [-1,1]
h = 0.1
h_val = np.arange(0, 2, h)

def derivative_calc(f, x, h, method):
    if method == "left":
        return (f(x) - f((x-h)))/h
        
    if method == "right":
        return (f((x+h))-f(x))/h
        
    if method == "central":
        return (f((x+h)) - f((x-h)))/(2*h)
    

x_val = []
left_diff = []
right_diff = []
central_diff = []

for x in range(int(x_range[0]/h), int(x_range[1]/h)):
    x_value = x*h
    left = derivative_calc(f, x_value, h, "left")
    right = derivative_calc(f, x_value, h, "right")
    central = derivative_calc(f, x_value, h, "central")
    
    x_val.append(x_value)
    left_diff.append(left)
    right_diff.append(right)
    central_diff.append(central)
    






h_val2 = np.arange(0.01, 1.01, 0.01)


mae_left = []
mae_right = []
mae_central = []

for h in h_val2:
    
    x_val2 = []
    left_diff2 = []
    right_diff2 = []
    central_diff2 = []


    for x in range(int(-1/h), int(1/h)):
        x_value = x*h
        left = abs(derivative_calc(f, x_value, h, "left") - exact_value(x_value))
        right = abs(derivative_calc(f, x_value, h, "right") - exact_value(x_value))
        central = abs(derivative_calc(f, x_value, h, "central") - exact_value(x_value))

        x_val2.append(x_value)
        left_diff2.append(left)
        right_diff2.append(right)
        central_diff2.append(central)

    mae_left.append(mean(left_diff2))
    mae_right.append(mean(right_diff2))
    mae_central.append(mean(central_diff2))



fig, axs = plt.subplots(1, 2, figsize=(15, 5))


axs[0].plot(x_val, left_diff, label = "left")
axs[0].plot(x_val, right_diff, label = "right")
axs[0].plot(x_val, central_diff, label = "central")



axs[1].plot(h_val2, mae_left, label = "left error")
axs[1].plot(h_val2, mae_right, label = "right error")
axs[1].plot(h_val2, mae_central, label = "central error")


axs[0].set_title("Wykres pochodnej (h = 0.1)")
axs[0].set_ylabel("Wartość pochodnej")
axs[0].set_xlabel("$x$")
axs[0].legend()

axs[1].set_title("Wykres błędu względnego")
axs[1].set_ylabel("Błąd względny")
axs[1].set_xlabel("$h$")
axs[1].legend()



plt.legend()
plt.show()