"""Compute and plot a Poincare map of a chaotic
Duffing oscillator. The script is mainly written
to exploit capabilities of a cheap GPU with CUDA.

When I run it on my CPU (i5-4570) it takes around 430 s.
This graphic card cost today ~100 EUR (17.01.2021).
"""
import numpy as np
import matplotlib.pyplot as plt
import time


def solve_ode(x, time):
    """
    Solve 2DoF ode on gpu, given
    the initial conditions. The
    result will be stored in the
    input array.
    

    Parameters
    ----------
    x : np.array 2D
        Contains initial conditions for the simulations.
        The elements are arranged in pairs:
        [[x1_sim1, x2_sim1], [x1_sim2, x2_sim2], ...]
    time : np.array
        Three-element list with time details.

    Returns
    -------
    x : np.array 2D

    """
    t0 = time[0]
    t_end = time[1]
    dt = time[2]

    for t in np.arange(t0, t_end, dt):
        dxdt0 = x[1]
        dxdt1 = 10*np.sin(t) - 0.1*x[1] - x[0]**3

        x[0] += dxdt0 * dt
        x[1] += dxdt1 * dt

    return x

simulations = 1000

# time of simulation
t0 = 0
t_end = 100
dt = 0.01
t = [t0, t_end, dt]

# generate initial conditions
x0 = np.random.random_sample((simulations, 2))

# simulate
start = time.perf_counter()
x = map(solve_ode, x0)
end = time.perf_counter()
print(end-start)

# plot results
x = list(x)
x = np.array(x)
plt.scatter(x[:, 0], x[:, 1], s=1)
plt.show()
