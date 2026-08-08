# KSL Procedure #
import numpy as np
import matplotlib.pyplot as plt

# Parameters #
Rank = 5
nx, nv = (25, 25)
x = np.linspace(0, 1, nx)
v = np.linspace(0, 1, nv)
dx = (x[-1]-x[0])/(nx-1)
dv = (v[-1]-v[0])/(nv-1)
xm, vm = np.meshgrid(x, v, indexing='ij')
f_0 = (1 + 0.01*np.cos(5*xm))*np.cos(2*np.pi*vm - 1)
T_span = [0, 20]
dT = 0.005
nt = int((T_span[1] - T_span[0])/dT)

# Finite difference matrices #
from diff_mat import *
D_x = central_Dif(nx, dx, True)
D_v = central_Dif(nv, dv, True)

# ODEs, Electric Field, and Local RK4 Method #
from equations import *

# Truncate initial condition down to the given rank #
U0, S0, Vh0 = np.linalg.svd(f_0, full_matrices=False)
S0 = np.diag(S0)
U0_low = U0[:, :Rank]
S0_low = S0[:Rank, :Rank]
Vh0_low = Vh0[:Rank, :]

# Main loop for the KSL Procedure #
U, S, Vh = U0_low, S0_low, Vh0_low
U_curr, S_curr, Vh_curr = U, S, Vh
for T in range(nt):
    # K-Step
    def DtK(Kk):
        E = Electric_Field(Kk @ Vh_curr, x, v)
        return K_del_t(Kk, Vh_curr, v, E, dx, dv, D_x, D_v)
    K_next = rk4(DtK, U_curr @ S_curr, dT)
    U_next, S_star = np.linalg.qr(K_next)

    # S-Step
    def DtS(Sk):
        E = Electric_Field(U_next @ Sk @ Vh_curr, x, v)
        return -S_del_t(U_next, Sk, Vh_curr, v, E, dx, dv, D_x, D_v)
    S_star_star = rk4(DtS, S_star, -dT)

    # L-Step
    def DtL(Lk):
        E = Electric_Field(U_next @ Lk, x, v)
        return L_del_t(U_next, Lk, v, E, dx, dv, D_x, D_v)
    L_next = rk4(DtL, S_star_star @ Vh_curr, dT)
    Vh_next, S_next = np.linalg.qr(L_next.T)

    # Update current decomposition
    U_curr, S_curr, Vh_curr = U_next, S_next.T, Vh_next.T

# Construct the final state of the system #
F = U_curr @ S_curr @ Vh_curr
plt.figure()
plt.scatter(xm, vm)
cp = plt.pcolormesh(xm, vm, F, cmap='viridis')
plt.colorbar(cp)
plt.show()
