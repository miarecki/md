import numpy as np
import matplotlib.pyplot as plt

# Model FitzHugh-Nagumo

def FN_model(x, y, R, I, a, b, tau):
    dxdt = x - (x**3)/3 - y + R*I
    dydt = (x + a - b*y)/tau
    return dxdt, dydt

# Runge - Kutta :: stopien 2 (RK2) dla modelu FN

def RK2(h, x0, y0, R, I, a, b, tau, num_steps):
    results = np.zeros((num_steps + 1, 2))
    results[0, :] = x0, y0

    for i in range(num_steps):
        x, y = results[i, :]
        k1x, k1y = FN_model(x, y, R, I, a, b, tau)
        k2x, k2y = FN_model(x + h * k1x, y + h * k1y, R, I, a, b, tau)

        x_new = x + 0.5 * h * (k1x + k2x)
        y_new = y + 0.5 * h * (k1y + k2y)

        results[i + 1, :] = x_new, y_new

    return results

# Runge - Kutta :: stopien 4 (RK4) dla modelu FN

def RK4(h, x0, y0, R, I, a, b, tau, num_steps):
    results = np.zeros((num_steps + 1, 2))
    results[0, :] = x0, y0

    for i in range(num_steps):
        x, y = results[i, :]
        
        k1x, k1y = FN_model(x, y, R, I, a, b, tau)
        k2x, k2y = FN_model(x + 0.5 * h * k1x, y + 0.5 * h * k1y, R, I, a, b, tau)
        k3x, k3y = FN_model(x + 0.5 * h * k2x, y + 0.5 * h * k2y, R, I, a, b, tau)
        k4x, k4y = FN_model(x + h * k3x, y + h * k3y, R, I, a, b, tau)

        x_new = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        results[i + 1, :] = x_new, y_new

    return results

if __name__ == '__main__':

    # Parameters
    a = 0.7
    tau = 12.5
    R = 0.1

    # Initial conditions
    x0 = -2  # Initial prey population
    y0 = -2  # Initial predator population

    I = 5.37
    b = 2

    # Time step and number of steps
    h = 0.1
    num_steps = 100

    # Run the RK2 method
    solution = RK2(h, x0, y0, R, I, a, b, tau, num_steps)

    # Plot the results
    time_values = np.linspace(0, num_steps * h, num_steps + 1) 
    plt.plot(time_values, solution[:, 0], label='x')
    plt.plot(time_values, solution[:, 1], label='y')
    plt.xlabel('T')
    plt.ylabel(':)')
    plt.legend()
    plt.title('FitzHugh-Nagumo (RK2)')
    plt.show()

    # Run the RK4 method
    solution = RK4(h, x0, y0, R, I, a, b, tau, num_steps)

    # Plot the results
    time_values = np.linspace(0, num_steps * h, num_steps + 1) 
    plt.plot(time_values, solution[:, 0], label='x')
    plt.plot(time_values, solution[:, 1], label='y')
    plt.xlabel('T')
    plt.ylabel(':)')
    plt.legend()
    plt.title('FitzHugh-Nagumo (RK4)')
    plt.show()

