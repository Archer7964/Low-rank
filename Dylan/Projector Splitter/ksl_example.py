# Externel Packages #
import matplotlib.pyplot as plt
import numpy as np

# Rank Selection (Archer) #
from helpers import *

# Main loop for the KSL Procedure #
def main():
    # Parameters #
    MAX_RANK = 4
    RANK_TOL = 1e-10
    nx, nv = (50, 50)
    x = (np.arange(nx) + 0.5) / nx
    v = (np.arange(nv) + 0.5) / nv
    f_0_x = 1.0 + 0.2 * np.cos(2.0 * np.pi * x)
    f_0_v = 1.0 + 0.2 * np.cos(2.0 * np.pi * v)
    a = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    b = 1.0 + 0.2 * np.sin(2.0 * np.pi * v)
    f_0 = f_0_x[:, None] @ f_0_v[None, :]
    T_span = [0, 1]
    dT = 0.01
    nt = int((T_span[1] - T_span[0]) // dT)

    # Truncate initial condition down to the given rank #
    U0, singular_values, Vh0 = np.linalg.svd(f_0, full_matrices=False)
    S0 = np.diag(singular_values)

    U_curr, S_curr, Vh_curr = U0, S0, Vh0
    ranks = []
    for T in range(nt):
        # K-Step
        def DtK(K): # xxr
            proj_VV = Vh_curr @ Vh_curr.T # rxr
            proj_bV = b[None, :] @ Vh_curr.T # 1xr
            return -K @ proj_VV.T + a[:, None] @ proj_bV
        K_next = rk4(DtK, U_curr @ S_curr, dT)
        U_next, S_star = np.linalg.qr(K_next)

        # S-Step
        def DtS(S): # rxr
            proj_UU = U_next.T @ U_next # rxr
            proj_aU = U_next.T @ a[:, None] # rx1
            proj_VV = Vh_curr @ Vh_curr.T # rxr
            proj_bV = b[None, :] @ Vh_curr.T # 1xr
            return -proj_UU @ S @ proj_VV.T + proj_aU @ proj_bV
        S_star_star = rk4(DtS, S_star, -dT)

        # L-Step
        def DtL(L): # rxv
            proj_UU = U_next.T @ U_next  # rxr
            proj_aU = U_next.T @ a[:, None]  # rx1
            return -proj_UU @ L + proj_aU @ b[None, :]
        L_next = rk4(DtL, S_star_star @ Vh_curr, dT)
        Vh_next, S_next = np.linalg.qr(L_next.T)

        # Update current decomposition
        U, singular_values, Vh = np.linalg.svd(U_next@S_next.T@Vh_next.T, full_matrices=False)
        rank = select_rank(singular_values, MAX_RANK, RANK_TOL)
        ranks.append(rank)
        U_curr, S_curr, Vh_curr = U[:, :rank], np.diag(singular_values[:rank]), Vh[:rank, :]

    # Construct the final state of the system #
    F = U_curr @ S_curr @ Vh_curr

    # Exact solution for t=1
    F_exact = f_0*np.exp(-1) + (1 - np.exp(-1))*(a[:, None]@b[None,:])

    # error analysis
    Error = np.linalg.norm(F_exact - F)
    print(Error)

    # Compare initial condition to the final state
    xm, vm = np.meshgrid(x, v)

    plt.subplot(2,3,1)
    plt.title("IC (t=0)")
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, f_0/np.max(f_0), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(cp)
    plt.subplot(2,3,2)
    plt.title("Aprox (t=1)")
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, F/np.max(f_0), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(cp)
    plt.subplot(2, 3, 3)
    plt.title("Exact (t=1)")
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, F_exact/np.max(f_0), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(cp)
    plt.subplot(2, 3, 5)
    plt.title("Error (t=1) (order 1e-2)")
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, np.abs(F_exact-F.T)/np.max(f_0), cmap='viridis', vmin=0, vmax=0.01)
    plt.colorbar(cp)
    plt.show()

    # print(ranks)

if __name__ == '__main__':
    main()
