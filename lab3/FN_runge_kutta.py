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
