import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# SIR Model with births and deaths

def SIR_model(S, I, R, m, N, b, g):
    dsdt = m * N - b * S * I - m * S
    didt = b * S * I - g * I - m * I
    drdt = g * I - m * R
    return dsdt, didt, drdt

# Runge-Kutte (RK4) for SIR

def RK4(h, S0, I0, m, N, b, g, num_steps):
    R0 = 0
    results = np.zeros((num_steps + 1, 3))
    results[0, :] = S0, I0, R0

    for i in range(num_steps):
        S, I, R = results[i, :]

        k1S, k1I, k1R = SIR_model(S, I, R, m, N, b[i], g)
        k2S, k2I, k2R = SIR_model(S + 0.5 * h * k1S, I + 0.5 * h * k1I, R + 0.5 * h * k1R, m, N, b[i], g)
        k3S, k3I, k3R = SIR_model(S + 0.5 * h * k2S, I + 0.5 * h * k2I, R + 0.5 * h * k2R, m, N, b[i], g)
        k4S, k4I, k4R = SIR_model(S + h * k3S, I + h * k3I, R + h * k3R, m, N, b[i], g)

        S_new = S + (h / 6.0) * (k1S + 2 * k2S + 2 * k3S + k4S)
        I_new = I + (h / 6.0) * (k1I + 2 * k2I + 2 * k3I + k4I)
        R_new = R + (h / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)

        results[i + 1, 0] = S_new
        results[i + 1, 1] = I_new
        results[i + 1, 2] = R_new

    return results

# Parameters
m = 10  # Birth and death rate
N = 1000  # Total population
h = 0.0001
num_steps = 365 * 2  # Number of days
g = 50

# Initial conditions
S0 = 900
I0 = 100
b = 1 # Original transmission rate

# Scenarios
scenarios = {
    'Miękki lockdown': {'start_day': 0, 'lockdown_duration': 30},
    'Umiarkowany Lockdown': {'start_day': 0, 'lockdown_duration': 90},
    'Twardy Lockdown': {'start_day': 0, 'lockdown_duration': 180}
}

# Store the total number of infections for each scenario
total_infections = {}

# Create subplots
num_subplots = len(scenarios)
fig, axes = plt.subplots(1,num_subplots, figsize=(15, 5), sharex=True)


# Initialize b_values with the original value
for i, (scenario, params) in enumerate(scenarios.items()):
    lines_begin = []
    lines_end = []
    start_day = params['start_day']
    lockdown_duration = params['lockdown_duration']

    # Implement lockdown scenarios
    test = []
    for day in range(start_day, lockdown_duration + 1):
        test.append(0.25 * b + 0.75 * b * (day / lockdown_duration))

    # Two weeks period between next lockdown
    for _ in range(14):
        test.append(b)

    b_values = test * 40
    b_values = b_values[:num_steps + 1]

    # Run the model for each scenario
    results = RK4(h, S0, I0, m, N, b_values, g, num_steps)

    # Store the total number of infections for the scenario
    total_infections[scenario] = np.sum(results[:, 1])

    # Define lines_begin and lines_end dynamically
    potential_lines_begin = [start_day + k * (lockdown_duration + 14) for k in range(100)]
    potential_lines_end = [start_day + k * (lockdown_duration + 14) + lockdown_duration - 1 for k in range(100)]

    # Cut the lists so that the maximum value is 730
    lines_begin.extend([x for x in potential_lines_begin if x <= 730])
    lines_end.extend([x for x in potential_lines_end if x <= 730])

    # Plot the results on a separate subplot
    axes[i].plot(results[:, 1], label=scenario)
    axes[i].vlines(lines_begin, ymin = 0, ymax = 1000, colors = 'green', linewidth = 0.6, linestyle = 'dashed')
    axes[i].vlines(lines_end, ymin = 0, ymax = 1000, colors = 'red', linewidth = 0.6, linestyle = 'dashed')
    axes[i].set_title(scenario)
    axes[i].set_ylabel('Liczba zarażonych')

    # Set x-ticks and labels for the last subplot
    today = datetime.today()
    x_ticks = np.arange(0, num_steps + 1, 100)
    x_labels = [(today + timedelta(days=int(x))).strftime('%Y-%m-%d') for x in x_ticks]

    axes[i].set_xticks(x_ticks)
    axes[i].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[i].set_xlabel('Czas (dni)')


plt.tight_layout()
plt.show()

# Print the total number of infections for each scenario
for scenario, total_infected in total_infections.items():
    print(f'{scenario}: {total_infected} wszystkich infekcji')