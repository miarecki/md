import numpy as np
import matplotlib.pyplot as plt

# Lorenz Model 

def LorenzModel(x, y, z, sigma, r, q):
    dxdt = sigma*(y-x)
    dydt = r*x-y-x*z
    dzdt = x*y - q*z
    return dxdt, dydt, dzdt

# Runge - Kutte :: (RK2) for Lorenz

def RK2(h, x0, y0, z0, sigma, r, q, num_steps):
    results = np.zeros((num_steps + 1, 3))
    results[0, :] = x0, y0, z0

    for i in range(num_steps):
        x, y, z = results[i, :]
        k1x, k1y, k1z = LorenzModel(x, y, z, sigma, r, q)
        k2x, k2y, k2z = LorenzModel(x + h * k1x, y + h * k1y, z + h*k1z, sigma, r, q)

        x_new = x + 0.5 * h * (k1x + k2x)
        y_new = y + 0.5 * h * (k1y + k2y)
        z_new = z + 0.5 * h * (k1z + k2z)

        results[i + 1, :] = x_new, y_new, z_new

    return results


# Runge - Kutte :: (RK4) for Lorenz

def RK4(h, x0, y0, z0, sigma, r, q, num_steps):
    results = np.zeros((num_steps + 1, 3))
    results[0, :] = x0, y0, z0

    for i in range(num_steps):
        x, y, z = results[i, :]
        
        k1x, k1y, k1z = LorenzModel(x, y, z, sigma, r, q)
        k2x, k2y, k2z = LorenzModel(x + 0.5 * h * k1x, y + 0.5 * h * k1y, z + 0.5 * h * k1z, sigma, r, q)
        k3x, k3y, k3z = LorenzModel(x + 0.5 * h * k2x, y + 0.5 * h * k2y, z + 0.5 * h * k2z, sigma, r, q)
        k4x, k4y, k4z = LorenzModel(x + h * k3x, y + h * k3y, z + h * k3z, sigma, r, q)

        x_new = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_new = y + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z_new = z + (h / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)


        results[i + 1, :] = x_new, y_new, z_new

    return results

if __name__ == '__main__':
        
    # Parameters

    # Initial conditions
    x0, y0, z0 = 1, 1, 1
    sigma = 10.0
    r = 28.0
    q = 8/3

    # Time step and number of steps
    h = 0.01
    num_steps = 10000

    # :)

    # Run RK2 and RK4 methods
    results_rk2 = RK2(h, x0, y0, z0, sigma, r, q, num_steps)
    results_rk4 = RK4(h, x0, y0, z0, sigma, r, q, num_steps)

    # Plotting
    fig = plt.figure(figsize=(15, 6))

    # Plotting RK2 results in 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(results_rk2[:, 0], results_rk2[:, 1], results_rk2[:, 2], label='RK2', color='blue')
    ax1.set_title('RK2 - Lorenz Model')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Plotting RK4 results in 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(results_rk4[:, 0], results_rk4[:, 1], results_rk4[:, 2], label='RK4', color='red')
    ax2.set_title('RK4 - Lorenz Model')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    plt.tight_layout()
    plt.show()
