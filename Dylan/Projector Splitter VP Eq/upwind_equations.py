# Externel Packages #
import numpy as np

# Equations (9)-(11) from paper #
# Upwind difference scheme #

def K_del_t(K_tx, Vh_tv, vel, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg):
    # Create positive/negative vel and E
    v_pos = np.maximum(vel, 0)
    v_neg = np.minimum(vel, 0)
    E_pos = np.maximum(E, 0)
    E_neg = np.minimum(E, 0)
    # Compute positive/negative derivatives
    K_del_x_pos = D_x_pos @ K_tx
    K_del_x_neg = D_x_neg @ K_tx
    Vh_del_v_pos = Vh_tv @ D_v_pos.T
    Vh_del_v_neg = Vh_tv @ D_v_neg.T
    # Compute positive/negative projections
    proj_VvV_pos = Vh_tv @ (v_pos * Vh_tv).T
    proj_VvV_neg = Vh_tv @ (v_neg * Vh_tv).T
    proj_VdV_pos = Vh_tv @ Vh_del_v_pos.T
    proj_VdV_neg = Vh_tv @ Vh_del_v_neg.T
    # Combine v terms and E terms into one
    v_sol = -K_del_x_neg @ proj_VvV_neg.T - K_del_x_pos @ proj_VvV_pos.T
    E_sol = np.diag(E_neg) @ K_tx @ proj_VdV_neg.T + np.diag(E_pos) @ K_tx @ proj_VdV_pos.T

    return v_sol + E_sol

def S_del_t(U_tx, S_t, Vh_tv, vel, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg):
    # Create positive/negative vel and E
    v_pos = np.maximum(vel, 0)
    v_neg = np.minimum(vel, 0)
    E_pos = np.maximum(E, 0)
    E_neg = np.minimum(E, 0)
    # Compute positive/negative derivatives
    U_del_x_pos = D_x_pos @ U_tx
    U_del_x_neg = D_x_neg @ U_tx
    Vh_del_v_pos = Vh_tv @ D_v_pos.T
    Vh_del_v_neg = Vh_tv @ D_v_neg.T
    # Compute positive/negative projections
    proj_UdU_pos = U_tx.T @ U_del_x_pos
    proj_UdU_neg = U_tx.T @ U_del_x_neg
    proj_UEU_pos = U_tx.T @ np.diag(E_pos) @ U_tx
    proj_UEU_neg = U_tx.T @ np.diag(E_neg) @ U_tx
    proj_VvV_pos = Vh_tv @ (v_pos * Vh_tv).T
    proj_VvV_neg = Vh_tv @ (v_neg * Vh_tv).T
    proj_VdV_pos = Vh_tv @ Vh_del_v_pos.T
    proj_VdV_neg = Vh_tv @ Vh_del_v_neg.T
    # Combine v terms and E terms into one
    v_sol = -proj_UdU_neg @ S_t @ proj_VvV_neg.T - proj_UdU_pos @ S_t @ proj_VvV_pos.T
    E_sol = proj_UEU_neg @ S_t @ proj_VdV_neg.T + proj_UEU_pos @ S_t @ proj_VdV_pos.T

    return v_sol + E_sol

def L_del_t(U_tx, L_tv, vel, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg):
    # Create positive/negative vel and E
    v_pos = np.maximum(vel, 0)
    v_neg = np.minimum(vel, 0)
    E_pos = np.maximum(E, 0)
    E_neg = np.minimum(E, 0)
    # Compute positive/negative derivatives
    U_del_x_pos = D_x_pos @ U_tx
    U_del_x_neg = D_x_neg @ U_tx
    L_del_v_pos = L_tv @ D_v_pos.T
    L_del_v_neg = L_tv @ D_v_neg.T
    # Compute positive/negative projections
    proj_UdU_pos = U_tx.T @ U_del_x_pos
    proj_UdU_neg = U_tx.T @ U_del_x_neg
    proj_UEU_pos = U_tx.T @ np.diag(E_pos) @ U_tx
    proj_UEU_neg = U_tx.T @ np.diag(E_neg) @ U_tx
    # Combine v terms and E terms into one
    v_sol = -proj_UdU_neg @ L_tv * v_neg - proj_UdU_pos@ L_tv * v_pos
    E_sol = proj_UEU_neg @ L_del_v_neg + proj_UEU_pos @ L_del_v_pos

    return v_sol + E_sol


from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson

# Equation for electric field E(x, t) #

def Electric_Field(phase_density, x, v):
    rho = 1 - simpson(phase_density, x=v, axis=1)
    E = -cumulative_simpson(rho, x=x, initial=0)
    return E - np.mean(E)
