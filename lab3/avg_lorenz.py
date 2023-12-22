import numpy as np
import matplotlib.pyplot as plt
from Lorenz_runge_kutta import LorenzModel, RK2, RK4



# ===========================================================================================================

# srednia różnica między RK2(h) i RK2(h/2) oraz RK4(h) i RK4(h/2)

# ===========================================================================================================



def calculate_average_difference(h, num_steps):
    # Run RK2 and RK4 methods with step size h and 0.5 * h
    results_rk2_h = RK2(h, x0, y0, z0, sigma, r, q, num_steps)
    results_rk2_half_h = RK2(0.5 * h, x0, y0, z0, sigma, r, q, 2 * num_steps)

    results_rk4_h = RK4(h, x0, y0, z0, sigma, r, q, num_steps)
    results_rk4_half_h = RK4(0.5 * h, x0, y0, z0, sigma, r, q, 2 * num_steps)

    # Calculate average differences
    avg_diff_rk2 = np.mean(np.abs(results_rk2_h - results_rk2_half_h[::2, :]))
    avg_diff_rk4 = np.mean(np.abs(results_rk4_h - results_rk4_half_h[::2, :]))

    return avg_diff_rk2, avg_diff_rk4

# Parameters
x0, y0, z0 = 1, 1, 1
sigma = 10.0
r = 28.0
q = 8/3
num_steps = 1000

# List of step sizes to compare
h_min, h_max, h_step = 0.01, 0.05, 0.001
h_values = np.arange(h_min, h_max, h_step)

# Calculate average differences for each step size
avg_diff_rk2_values = []
avg_diff_rk4_values = []

for h in h_values:
    avg_diff_rk2, avg_diff_rk4 = calculate_average_difference(h, num_steps)
    avg_diff_rk2_values.append(avg_diff_rk2)
    avg_diff_rk4_values.append(avg_diff_rk4)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(h_values, avg_diff_rk2_values, label='RK2', marker='o')
plt.plot(h_values, avg_diff_rk4_values, label='RK4', marker='o')

plt.xlabel('Krok h > 0')
plt.ylabel('Średnia różnica')
plt.legend()
plt.grid(True)
plt.show()



# ===========================================================================================================

# MAE dla x, y, z jak w 
# https://colab.research.google.com/drive/1O4FO7O4bUhzvYZMyo09kIhvWj3a_6CPb?usp=sharing#scrollTo=Jjl4-LgOLTTS
# dla krolikow i lisow/

# ===========================================================================================================



# Error analysis for Lorenz model
sigma = 10.0
r = 28.0
q = 8/3
h_min, h_max, h_step = 0.01, 0.05, 0.001
num_steps = 1000


rk2_error_x, rk2_error_y, rk2_error_z = [], [], []
rk4_error_x, rk4_error_y, rk4_error_z = [], [], []


for h in np.arange(h_min, h_max, h_step):
    # RK2 method relative error
	result_prev = RK2(h, 1, 1, 1, sigma, r, q, num_steps)
	result_next = RK2(h/2, 1, 1, 1, sigma, r, q, num_steps)
	result_next = result_next[::2, :] 

	x_prev, y_prev, z_prev = result_prev[:, 0], result_prev[:, 1], result_prev[:, 2]
	x_next, y_next, z_next = result_next[:, 0], result_next[:, 1], result_next[:, 2]

	# Adjust lengths
	min_len = min(len(x_next), len(x_prev))
	x_prev, x_next = x_prev[:min_len], x_next[:min_len]
	y_prev, y_next = y_prev[:min_len], y_next[:min_len]
	z_prev, z_next = z_prev[:min_len], z_next[:min_len]

	rk2_error_x.append(np.mean(np.abs(x_next - x_prev)))
	rk2_error_y.append(np.mean(np.abs(y_next - y_prev)))
	rk2_error_z.append(np.mean(np.abs(z_next - z_prev)))

	# RK4 method relative error
	result_prev = RK4(h, 1, 1, 1, sigma, r, q, num_steps)
	result_next = RK4(h/2, 1, 1, 1, sigma, r, q, num_steps)
	result_next = result_next[::2, :]  

	x_prev, y_prev, z_prev = result_prev[:, 0], result_prev[:, 1], result_prev[:, 2]
	x_next, y_next, z_next = result_next[:, 0], result_next[:, 1], result_next[:, 2]

	# Adjust lengths
	min_len = min(len(x_next), len(x_prev))
	x_prev, x_next = x_prev[:min_len], x_next[:min_len]
	y_prev, y_next = y_prev[:min_len], y_next[:min_len]
	z_prev, z_next = z_prev[:min_len], z_next[:min_len]

	rk4_error_x.append(np.mean(np.abs(x_next - x_prev)))
	rk4_error_y.append(np.mean(np.abs(y_next - y_prev)))
	rk4_error_z.append(np.mean(np.abs(z_next - z_prev)))

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Error as a function of step")

axs[0].plot(np.arange(h_min, h_max, h_step), rk2_error_x, label="RK2")
axs[0].plot(np.arange(h_min, h_max, h_step), rk4_error_x, label="RK4")
axs[0].set_xlabel("$h$")
axs[0].set_ylabel("MAE")
axs[0].set_title("MAE dla x")
axs[0].legend()

axs[1].plot(np.arange(h_min, h_max, h_step), rk2_error_y, label="RK2")
axs[1].plot(np.arange(h_min, h_max, h_step), rk4_error_y, label="RK4")
axs[1].set_xlabel("$h$")
axs[1].set_ylabel("MAE")
axs[1].set_title("MAE dla y")
axs[1].legend()

axs[2].plot(np.arange(h_min, h_max, h_step), rk2_error_z, label="RK2")
axs[2].plot(np.arange(h_min, h_max, h_step), rk4_error_z, label="RK4")
axs[2].set_xlabel("$h$")
axs[2].set_ylabel("MAE")
axs[2].set_title("MAE dla z")
axs[2].legend()

plt.show()