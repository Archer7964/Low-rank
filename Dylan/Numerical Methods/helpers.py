# Externel Packages #
import numpy as np

# Rank Selection (Archer) #

def select_rank(singular_values, MAX_RANK, RANK_TOL):
    limit = RANK_TOL * np.linalg.norm(singular_values)
    for rank in range(1, len(singular_values) + 1):
        if np.linalg.norm(singular_values[rank:]) <= limit:
            return min(rank, MAX_RANK)
    return min(len(singular_values), MAX_RANK)

# Fixed-stepsize RK4 method that works on matrices to avoid reshaping #

def rk4(func, Y, dt):
    """
    Handles matrices for faster computation.
    :param func: ode
    :param Y: initial condition
    :param dt: time step
    :return: Returns the solution at time step t+dt.
    """
    k1 = func(Y)
    k2 = func(Y + 0.5*dt*k1)
    k3 = func(Y + 0.5*dt*k2)
    k4 = func(Y + dt*k3)
    return Y + (k1 + 2*k2 + 2*k3 + k4)*dt/6
