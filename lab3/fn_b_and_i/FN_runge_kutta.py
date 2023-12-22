import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

def update(val):
    b = slider_b.val
    I = slider_I.val
    solution = RK2(h, x0, y0, R, I, a, b, tau, num_steps)
    line.set_ydata(solution[:, 0])
    line2.set_ydata(solution[:, 1])
    fig.canvas.draw_idle()

if __name__ == '__main__':

    # Parameters
    a = 0.7
    tau = 12.5
    R = 2.5

    # Initial conditions
    x0 = 1  # Initial prey population
    y0 = 1  # Initial predator population

    b = 0.8
    I = 0.5

    # Time step and number of steps
    h = 0.001
    num_steps = 100000

    # Create sliders
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax_b = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_I = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider_b = Slider(ax=ax_b, label='b', valmin=0.1, valmax=2.0, valinit=b)
    slider_I = Slider(ax=ax_I, label='I', valmin=0.1, valmax=6, valinit=I)

    slider_b.on_changed(update)
    slider_I.on_changed(update)

    # Run the RK2 method
    solution = RK2(h, x0, y0, R, I, a, b, tau, num_steps)

    # Plot the results
    time_values = np.linspace(0, num_steps * h, num_steps + 1) 
    line, = ax.plot(time_values, solution[:, 0], label='v')
    line2, = ax.plot(time_values, solution[:, 1], label='w')
    ax.set_xlabel('t')
    ax.set_ylabel('v,w')
    ax.set_ylim([-8,8])
    ax.legend()
    ax.set_title(f'FitzHugh-Nagumo (RK2) (v(0), w(0)) = {x0, y0} ')
    plt.show()
