import numpy as np
import matplotlib.pyplot as plt

# FN model

# Parameters
a = 0.7
tau = 12.5
R = 2.5

# Initial conditions
x0 = 1  
y0 = 1  

I = .5
b = 0.8

# Time step and number of steps
h = 0.001
num_steps = 1000

f = lambda x, y: x - (x**3)/3 - y + R*I
g = lambda x, y: (x + a - b*y)/tau


# Function to build plotting grids
def build_plotting_grids(f, g, x_values, y_values):
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x_prime = f(x_grid, y_grid)
    y_prime = g(x_grid, y_grid)
    m = np.sqrt(x_prime**2 + y_prime**2)
    return x_grid, y_grid, x_prime/m, y_prime/m, m

# Runge-Kutta scheme for numerical integration
def rk_scheme(f, g, x0, y0, h, num_steps):
    x_values = [x0]
    y_values = [y0]

    for _ in range(num_steps):
        k1x = h * f(x_values[-1], y_values[-1])
        k1y = h * g(x_values[-1], y_values[-1])

        k2x = h * f(x_values[-1] + 0.5 * k1x, y_values[-1] + 0.5 * k1y)
        k2y = h * g(x_values[-1] + 0.5 * k1x, y_values[-1] + 0.5 * k1y)

        k3x = h * f(x_values[-1] + 0.5 * k2x, y_values[-1] + 0.5 * k2y)
        k3y = h * g(x_values[-1] + 0.5 * k2x, y_values[-1] + 0.5 * k2y)

        k4x = h * f(x_values[-1] + k3x, y_values[-1] + k3y)
        k4y = h * g(x_values[-1] + k3x, y_values[-1] + k3y)

        x_new = x_values[-1] + (1 / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y_values[-1] + (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)

        x_values.append(x_new)
        y_values.append(y_new)

    return np.array(x_values), np.array(y_values)

# Function to plot trajectories
def plot_trajectories(f, g, x_values, y_values, h, num_steps, color='black'):
    for x_0 in np.arange(-5, 5, .5):
        for y_0 in np.arange(-5, 5, .5):
            x, y = rk_scheme(f, g, x_0, y_0, h, num_steps)
            plt.plot(x, y, color=color, linewidth=.2)

# Plotting grids, nullclines, and trajectories
x_values = np.arange(-5, 5, 0.5)
y_values = np.arange(-5,5,1)
x_grid, y_grid, x_prime, y_prime, m = build_plotting_grids(f, g, x_values, y_values)

plt.quiver(x_grid, y_grid, x_prime, y_prime, m)
plt.plot(x_values, [x - (x**3)/3 + R*I for x in x_values], label='Nullcline for v\' = 0')
plt.ylim([-5, 5])
plt.plot(y_values, [(y+a)/b for y in y_values], label='Nullcline for w\' = 0')
plt.ylim([-5, 5])
plot_trajectories(f, g, x_values, y_values, 0.1, 1000)
plt.xlim([-2,2])

# Set labels and title

plt.xlabel('x')
plt.ylabel('y')

plt.title('Phase Portrait, Nullclines, and Trajectories')

# Add legend
plt.legend()

# Show the plot
plt.show()
