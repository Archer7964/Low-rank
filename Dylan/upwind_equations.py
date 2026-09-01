# Externel Packages #
import numpy as np

# Equations (9)-(11) from paper #
# Upwind difference scheme #

def K_del_t(K_tx, Vh_tv, vel, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg):
    # positive/negative vel and E effects the difference mode
    v_pos = np.clip(vel, 0, 10^6)
    v_neg = np.clip(vel, -10^6, 0)
    E_pos = np.clip(E, 0, 10 ^ 6)
    E_neg = np.clip(E, -10 ^ 6, 0)

    K_del_x_pos = D_x_pos @ K_tx
    K_del_x_neg = D_x_neg @ K_tx
    Vh_del_v_pos = Vh_tv @ D_v_pos.T
    Vh_del_v_neg = Vh_tv @ D_v_neg.T

    # v+ & E+
    proj_VvV = Vh_tv @ (v_pos * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_pos.T
    pos_pos_sol = -K_del_x_pos @ proj_VvV.T + np.diag(E_pos) @ K_tx @ proj_VdV.T

    # v+ & E-
    proj_VvV = Vh_tv @ (v_pos * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_neg.T
    pos_neg_sol = -K_del_x_pos @ proj_VvV.T + np.diag(E_neg) @ K_tx @ proj_VdV.T

    # v- & E+
    proj_VvV = Vh_tv @ (v_neg * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_pos.T
    neg_pos_sol = -K_del_x_neg @ proj_VvV.T + np.diag(E_pos) @ K_tx @ proj_VdV.T

    # v- & E-
    proj_VvV = Vh_tv @ (v_neg * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_neg.T
    neg_neg_sol = -K_del_x_neg @ proj_VvV.T + np.diag(E_neg) @ K_tx @ proj_VdV.T

    return pos_pos_sol + pos_neg_sol + neg_pos_sol + neg_neg_sol

def S_del_t(U_tx, S_t, Vh_tv, vel, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg):
    # positive/negative vel and E effects the difference mode
    v_pos = np.clip(vel, 0, 10 ^ 3)
    v_neg = np.clip(vel, -10 ^ 3, 0)
    E_pos = np.clip(E, 0, 10 ^ 3)
    E_neg = np.clip(E, -10 ^ 3, 0)

    U_del_x_pos = D_x_pos @ U_tx
    U_del_x_neg = D_x_neg @ U_tx
    Vh_del_v_pos = Vh_tv @ D_v_pos.T
    Vh_del_v_neg = Vh_tv @ D_v_neg.T

    # v+ & E+
    proj_UdU = U_tx.T @ U_del_x_pos
    proj_UEU = U_tx.T @ np.diag(E_pos) @ U_tx
    proj_VvV = Vh_tv @ (v_pos * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_pos.T
    pos_pos_sol = -proj_UdU @ S_t @ proj_VvV.T + proj_UEU @ S_t @ proj_VdV.T

    # v+ & E-
    proj_UdU = U_tx.T @ U_del_x_pos
    proj_UEU = U_tx.T @ np.diag(E_neg) @ U_tx
    proj_VvV = Vh_tv @ (v_pos * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_neg.T
    pos_neg_sol = -proj_UdU @ S_t @ proj_VvV.T + proj_UEU @ S_t @ proj_VdV.T

    # v- & E+
    proj_UdU = U_tx.T @ U_del_x_neg
    proj_UEU = U_tx.T @ np.diag(E_pos) @ U_tx
    proj_VvV = Vh_tv @ (v_neg * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_pos.T
    neg_pos_sol = -proj_UdU @ S_t @ proj_VvV.T + proj_UEU @ S_t @ proj_VdV.T

    # v+ & E-
    proj_UdU = U_tx.T @ U_del_x_neg
    proj_UEU = U_tx.T @ np.diag(E_neg) @ U_tx
    proj_VvV = Vh_tv @ (v_neg * Vh_tv).T
    proj_VdV = Vh_tv @ Vh_del_v_neg.T
    neg_neg_sol = -proj_UdU @ S_t @ proj_VvV.T + proj_UEU @ S_t @ proj_VdV.T

    return pos_pos_sol + pos_neg_sol + neg_pos_sol + neg_neg_sol

def L_del_t(U_tx, L_tv, vel, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg):
    # positive/negative vel and E effects the difference mode
    v_pos = np.clip(vel, 0, 10 ^ 3)
    v_neg = np.clip(vel, -10 ^ 3, 0)
    E_pos = np.clip(E, 0, 10 ^ 3)
    E_neg = np.clip(E, -10 ^ 3, 0)

    U_del_x_pos = D_x_pos @ U_tx
    U_del_x_neg = D_x_neg @ U_tx
    L_del_v_pos = L_tv @ D_v_pos.T
    L_del_v_neg = L_tv @ D_v_neg.T

    # v+ & E+
    proj_UdU = U_tx.T @ U_del_x_pos
    proj_UEU = U_tx.T @ np.diag(E_pos) @ U_tx
    pos_pos_sol = -proj_UdU @ L_tv * v_pos + proj_UEU @ L_del_v_pos

    # v+ & E-
    proj_UdU = U_tx.T @ U_del_x_pos
    proj_UEU = U_tx.T @ np.diag(E_neg) @ U_tx
    pos_neg_sol = -proj_UdU @ L_tv * v_pos + proj_UEU @ L_del_v_neg

    # v- & E+
    proj_UdU = U_tx.T @ U_del_x_neg
    proj_UEU = U_tx.T @ np.diag(E_pos) @ U_tx
    neg_pos_sol = -proj_UdU @ L_tv * v_neg + proj_UEU @ L_del_v_pos

    # v- & E+
    proj_UdU = U_tx.T @ U_del_x_neg
    proj_UEU = U_tx.T @ np.diag(E_neg) @ U_tx
    neg_neg_sol = -proj_UdU @ L_tv * v_neg + proj_UEU @ L_del_v_neg

    return pos_pos_sol + pos_neg_sol + neg_pos_sol + neg_neg_sol


from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson

# Equation for electric field E(x, t) #

def Electric_Field(phase_density, x, v):
    rho = 1 - simpson(phase_density, x=v, axis=1)
    E = -cumulative_simpson(rho, x=x, initial=0)
    return E - np.mean(E)
