"""Compute and plot a Poincare map of a chaotic
Duffing oscillator. The script is mainly written
to exploit capabilities of a cheap GPU with CUDA.

When I run it on my GPU (GT 710) it takes around 0.63s.
This graphic card cost today (17.01.2021) something like
one family visit to the cinema (~35EUR).
"""
import math
from numba import cuda, f4
import numpy as np
import matplotlib.pyplot as plt
import time


#NUMBA_ENABLE_CUDASIM = 1

@cuda.jit
def solver(x, time):
    t = time[0]
    t_end = time[1]
    dt = time[2]
    
    p = cuda.grid(1)
    
    dxdt = cuda.shared.array(2, f4)
    
    if p < x.size:
        while t < t_end:
            dxdt[0] = x[p, 1]
            dxdt[1] = np.float32(10)*math.sin(t) - np.float32(0.1)*x[p, 1] - x[p, 0]**3
            
            for i in range(2):
                x[p, i] += dxdt[i] * dt
                
            t += dt


dim = 2
inst = 10_000

t0 = 0
t_end = 100
dt = 0.01
t = np.array([t0, t_end, dt], dtype='float32')

x = np.random.random_sample((inst, dim)).astype('float32')

threads = inst
threads_per_block = 32*8
blocks_per_grid = threads // threads_per_block + 1

start = time.perf_counter()
solver[blocks_per_grid, threads_per_block](x, t)
print(f'Elapsed: {time.perf_counter() - start} [s]')

plt.scatter(x[:, 0], x[:, 1], s=1)

