# Externel Packages #
import matplotlib.pyplot as plt

# Finite difference matrices #
from diff_mat import *

# ODEs, Electric Field, and Local RK4 Method #
from equations import *

# Rank Selection (Archer) #
from helpers import *

# Main loop for the KSL Procedure #
def main():
    # Parameters #
    MAX_RANK = 15
    RANK_TOL = 1e-10
    nx, nv = (50, 50)
    x = np.linspace(0, 4*np.pi, nx)
    v = np.linspace(-6, 6, nv)
    dx = (x[-1] - x[0]) / (nx - 1)
    dv = (v[-1] - v[0]) / (nv - 1)
    # ex1
    f_0_x = np.exp(-0.2 * (x - 2 * np.pi) * (x - 2 * np.pi)) * np.sin(x / 2) * np.sin(x / 2)
    f_0_v = np.exp(-v * v / 2) / (2 * np.pi)
    # f_0_x = np.ones_like(x)
    # f_0_v = v*v * np.exp(-v * v / 2) / np.sqrt(2 * np.pi)
    perturb = np.diag(1 + 0.02*np.sin(0.5*x))
    f_0 = perturb @ f_0_x[:, None] * f_0_v[None, :]
    T_span = [0,1]
    dT = 0.01
    nt = int((T_span[1] - T_span[0]) / dT)
    print(nt)

    D_x_pos = upwind_Dif(nx, dx, True, True)
    D_x_neg = upwind_Dif(nx, dx, False, True)
    D_v_pos = upwind_Dif(nv, dv, True, True)
    D_v_neg = upwind_Dif(nx, dv, False, True)

    # Truncate initial condition down to the given rank #
    U0, singular_values, Vh0 = np.linalg.svd(f_0, full_matrices=False)
    S0 = np.diag(singular_values)

    # Store specific states
    F_1, F_2, F_3 = [[]],[[]],[[]]

    U_curr, S_curr, Vh_curr = U0, S0, Vh0
    ranks = []
    for T in range(nt):
        # Store specific states
        if T/nt == 0.3:
            F_1 = U_curr @ S_curr @ Vh_curr
        elif T/nt == 0.5:
            F_2 = U_curr @ S_curr @ Vh_curr
        elif T/nt == 0.7:
            F_3 = U_curr @ S_curr @ Vh_curr

        # K-Step
        def DtK(Kk):
            E = Electric_Field(Kk @ Vh_curr, x, v)
            return K_del_t(Kk, Vh_curr, v, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg)
        K_next = rk4(DtK, U_curr @ S_curr, dT)
        U_next, S_star = np.linalg.qr(K_next)

        # S-Step
        def DtS(Sk):
            E = Electric_Field(U_next @ Sk @ Vh_curr, x, v)
            return S_del_t(U_next, Sk, Vh_curr, v, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg)
        S_star_star = rk4(DtS, S_star, -dT)

        # L-Step
        def DtL(Lk):
            E = Electric_Field(U_next @ Lk, x, v)
            return L_del_t(U_next, Lk, v, E, D_x_pos, D_x_neg, D_v_pos, D_v_neg)
        L_next = rk4(DtL, S_star_star @ Vh_curr, dT)
        Vh_next, S_next = np.linalg.qr(L_next.T)

        # Update current decomposition
        U, singular_values, Vh = np.linalg.svd(U_next@S_next.T@Vh_next.T, full_matrices=False)
        rank = select_rank(singular_values, MAX_RANK, RANK_TOL)
        ranks.append(rank)
        U_curr, S_curr, Vh_curr = U[:, :rank], np.diag(singular_values[:rank]), Vh[:rank, :]

    # Construct the final state of the system #
    F = U_curr @ S_curr @ Vh_curr

    # Compare every stored state
    xm, vm = np.meshgrid(x, v)

    plt.subplot(1,4,1)
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, f_0, cmap='viridis')
    plt.colorbar(cp)
    plt.title("t=0.00")
    plt.subplot(1,4,2)
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, F_1, cmap='viridis')
    plt.colorbar(cp)
    plt.axis("off")
    plt.title("t=0.30")
    plt.subplot(1, 4, 3)
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, F_2, cmap='viridis')
    plt.colorbar(cp)
    plt.axis("off")
    plt.title("t=0.50")
    plt.subplot(1, 4, 4)
    plt.scatter(xm, vm)
    cp = plt.pcolormesh(xm, vm, F_3, cmap='viridis')
    plt.colorbar(cp)
    plt.axis("off")
    plt.title("t=0.60")
    plt.show()

    print(ranks)

if __name__ == '__main__':
    main()
