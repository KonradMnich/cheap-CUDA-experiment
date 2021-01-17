"""Compute and plot a Poincare map of a chaotic
Duffing oscillator. The script is mainly written
to exploit capabilities of a cheap GPU with CUDA.

When I run it on my GPU (GT 710) it takes around 0.63s.
This graphic card cost today (17.01.2021) something like
one family visit to the cinema (~35EUR).
"""
from numba import cuda # for parallel gpu computations
import numpy as np # for random samples and array management
import time # for timing
import math # for sinus
import matplotlib.pyplot as plt # for scatter plot


@cuda.jit
def solve_ode(x, time):
    """
    Solve 2DoF ode on gpu, given
    the initial conditions. The
    result will be stored in the
    input array.
    

    Parameters
    ----------
    x : np.array
        Contains initial conditions for the simulations.
        The elements are arranged in pairs:
        [x1_sim1, x2_sim1, x1_sim2, x2_sim2, ...]
    time : np.array
        Three-element list with time details.

    Returns
    -------
    None.

    """
    # time variables
    t = time[0]
    t_end = time[1]
    dt = time[2]
    
    # index of thread on GPU
    pos = cuda.grid(1)
    # mappping index to access every
    # second element of the array
    pos = pos * 2
    
    # condidion to avoid threads
    # accessing indices out of array
    if pos < x.size:
        # execute until the time reaches t_end
        while t < t_end:
            # compute derivatives
            dxdt0 = x[pos+1]
            dxdt1 = 10.0*math.sin(t) - 0.1*x[pos+1] - x[pos]**3
            
            # update state vecotr
            x[pos] += dxdt0 * dt
            x[pos+1] += dxdt1 * dt
            
            # update time
            t += dt
            
        
# number of independent oscillators
# to simulate
trials = 10_000

# time variables
t0 = 0
t_end = 100
dt = 0.01
t = np.array([t0, t_end, dt])

# generate random initial condiotions
init_states = np.random.random_sample(2 * trials)

# manage nr of threads (threads)
threads_per_block = 32
blocks_per_grid = \
    (init_states.size + (threads_per_block - 1)) // threads_per_block

# start timer
start = time.perf_counter()

# start parallel simulations
solve_ode[blocks_per_grid, threads_per_block](init_states, t)

# measure time elapsed
end = time.perf_counter()
print(f'The result was computed in {end-start} s')

# reshape the array into 2D
x = init_states.reshape((trials, 2))

# plot the phase space
plt.scatter(x[:, 0], x[:, 1], s=1)