# Externel Packages #
import numpy as np

# Equations (9)-(11) from paper #

def K_del_t(K_tx, Vh_tv, vel, E, D_x, D_v):
    K_del_x = D_x @ K_tx
    Vh_del_v = Vh_tv @ D_v.T

    proj_VvV = Vh_tv @ (vel * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v.T

    return -K_del_x @ proj_VvV.T + np.diag(E) @ K_tx @ proj_VdV.T

def S_del_t(U_tx, S_t, Vh_tv, vel, E, D_x, D_v):
    U_del_x = D_x @ U_tx
    Vh_del_v = Vh_tv @ D_v.T

    proj_UdU = U_tx.T @ U_del_x
    proj_UEU = U_tx.T @ np.diag(E) @ U_tx
    proj_VvV = Vh_tv @ (vel * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v.T

    return -proj_UdU @ S_t @ proj_VvV.T + proj_UEU @ S_t @ proj_VdV.T

def L_del_t(U_tx, L_tv, vel, E, D_x, D_v):
    U_del_x = D_x @ U_tx
    L_del_v = L_tv @ D_v.T

    proj_UdU = U_tx.T @ U_del_x
    proj_UEU = U_tx.T @ np.diag(E) @ U_tx

    return -proj_UdU @ L_tv * vel + proj_UEU @ L_del_v


from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson

# Equation for electric field E(x, t) #

def Electric_Field(phase_density, x, v):
    rho = 1 - simpson(phase_density, x=v, axis=1)
    E = -cumulative_simpson(rho, x=x, initial=0)
    return E - np.mean(E)
