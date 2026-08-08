import numpy as np

# Equations (9)-(11) from paper #

def K_del_t(K_tx, Vh_tv, vel, E, dx, dv, D_x, D_v):
    K_del_x = D_x @ K_tx
    Vh_del_v = Vh_tv @ D_v.T

    proj_VvV = Vh_tv @ (vel * Vh_tv).T * dv
    proj_VdV = Vh_tv @ Vh_del_v.T * dv

    return -K_del_x @ proj_VvV + np.diag(E) @ K_tx @ proj_VdV

def S_del_t(U_tx, S_t, Vh_tv, vel, E, dx, dv, D_x, D_v):
    U_del_x = D_x @ U_tx
    Vh_del_v = Vh_tv @ D_v.T

    proj_UdU = U_tx.T @ U_del_x * dx
    proj_UEU = U_tx.T @ np.diag(E) @ U_tx * dx
    proj_VvV = Vh_tv @ (vel * Vh_tv).T * dv
    proj_VdV = Vh_tv @ Vh_del_v.T * dv

    return -proj_UdU @ S_t @ proj_VvV + proj_UEU @ S_t @ proj_VdV

def L_del_t(U_tx, L_tv, vel, E, dx, dv, D_x, D_v):
    U_del_x = D_x @ U_tx
    L_del_v = L_tv @ D_v.T

    proj_UdU = U_tx.T @ U_del_x * dx
    proj_UEU = U_tx.T @ np.diag(E) @ U_tx * dx

    return -proj_UdU @ L_tv * vel + proj_UEU @ L_del_v
    #                          ^-- Should this velocity term be here?


from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson

# Equation for electric field E(x, t) #

def Electric_Field(phase_density, x, v):
    rho = 1 - simpson(phase_density, x=v, axis=1)
    E = -cumulative_simpson(rho, x=x, initial=0)
    return E - np.mean(E)


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
