import numba as nb
import numpy as np
import math


def solve_ode(x):
    dt = nb.f8(0.01)
    t = nb.f8(0)
    t_end = nb.f8(100)
    
    dxdt = nb.f8[:]
    
    for i in range(0, int(t_end), 2):
        while t < t_end:
            dxdt[0] = x[i]
            dxdt[1] = math.sin(t) - nb.f4(0.1) * x[i+1] - x[i]**3
            
            for j, dxdt_el in enumerate(dxdt):
                x[i] += dxdt_el * dt
            
            
samples = 10_000 * 2
x = np.random.random_sample(samples)
solve_ode(x)