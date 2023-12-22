import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# SIR Model with births and deaths

def SIR_model(S, I, R, m, N, b, g):
    dsdt = m*N - b*S*I - m*S
    didt = b*S*I - g*I - m*I
    drdt = g*I - m*R
    return dsdt, didt, drdt

# Runge - Kutte :: (RK2) for SIR

def RK2(h, S0, I0, m, N, b, g, num_steps):
    R0 = 0
    results = np.zeros((num_steps + 1, 3))
    results[0, :] = S0, I0, R0

    for i in range(num_steps):
        S, I, R = results[i, :]
        k1S, k1I, k1R = SIR_model(S, I, R, m, N, b, g)
        k2S, k2I, k2R = SIR_model(S + h * k1S, I + h * k1I, R + h*k1R, m, N, b, g)

        S_new = S + 0.5 * h * (k1S + k2S)
        I_new = I + 0.5 * h * (k1I + k2I)
        R_new = R + 0.5 * h * (k1R + k2R)

        results[i + 1, :] = S_new, I_new, R_new

    return results

# Runge - Kutte :: (RK4) for SIR

def RK4(h, S0, I0, m, N, b, g, num_steps):
    R0 = 0
    results = np.zeros((num_steps + 1, 3))
    results[0, :] = S0, I0, R0

    for i in range(num_steps):
        S, I, R = results[i, :]
        
        k1S, k1I, k1R = SIR_model(S, I, R, m, N, b, g)
        k2S, k2I, k2R = SIR_model(S + 0.5 * h * k1S, I + 0.5 * h * k1I, R + 0.5 * h * k1R, m, N, b, g)
        k3S, k3I, k3R = SIR_model(S + 0.5 * h * k2S, I + 0.5 * h * k2I, R + 0.5 * h * k2R, m, N, b, g)
        k4S, k4I, k4R = SIR_model(S + h * k3S, I + h * k3I, R + h * k3R, m, N, b, g)

        S_new = S + (h / 6.0) * (k1S + 2 * k2S + 2 * k3S + k4S)
        I_new = I + (h / 6.0) * (k1I + 2 * k2I + 2 * k3I + k4I)
        R_new = R + (h / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)


        results[i + 1, :] = S_new, I_new, R_new

    return results

if __name__ == '__main__':

    # Parameters
    m = 0.01  # Birth and death rate
    N = 1000  # Total population
    h = 0.0001
    num_steps = 365
    g = 0.1

    # Scenarios
    scenarios = [
        {"label": "Mało zarażeń, bardzo zaraźliwa", "b": 1.0, "I0": 5.0},
        {"label": "Dużo zarażeń, bardzo zaraźliwa", "b": 1.0, "I0": 50.0},
        {"label": "Mało zarażeń, mało zaraźliwa", "b": 0.2, "I0": 5.0},
        {"label": "Dużo zarażeń, mało zaraźliwa", "b": 0.2, "I0": 50.0},
    ]

    # Plotting
    plt.figure(figsize=(12, 8))

    for scenario in scenarios:
        S0, I0, b = N - scenario["I0"], scenario["I0"], scenario["b"]
        results_rk4 = RK4(h, S0, I0, m, N, b, g, num_steps)

        plt.plot(results_rk4[:, 1], label=f'{scenario["label"]}, $I_0$ = {I0}, beta = {b}')

    # Define x-ticks and labels
    today = datetime.today()
    x_ticks = np.arange(0, num_steps + 1, 50)
    x_labels = [(today + timedelta(days=int(x))).strftime('%Y-%m-%d') for x in x_ticks]

    # Set x-ticks and labels
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right')

    plt.title("Model SIR z przyrostem naturalnym")
    plt.xlabel("Czas")
    plt.ylabel("Liczba zarażonych")
    plt.legend()
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()