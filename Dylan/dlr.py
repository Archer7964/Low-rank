# KSL Procedure
# K-step: Integrate from t^n to t^n+1 of the PDE: Kj_del_t = Proj[f_del_t(KV^n), Vj], where Kj(t,x) = sum_i=1^r(Ui^n(t, x)Sij(t)
#           IC: U^n and S^n
#           Solve for K(t^n+1,x), then obtain U(t^n+1,x) = U^n+1 and S* from the QR decomposition of K^n+1
# S_step: Solve for S** backwards in time by integrating from t^n to t^n+1 of the PDE: Sij_del_t = -Proj[f_del_t(U^n+1 * S * V^n), Ui^n+1 * Vj^n]
#           TC: S*
#           Then S** = S*(t^n)
# L-step: Integrate from t^n to t^n+1 of the PDE Li_del_t = Proj[f_del_t(U^n+1L), Ui], where Li(t,v) = sum_j=1^r(Vj^n(t, v)Sij**(t)
#           IC: V^n and S**
#           Solve for L(t^n+1,v), then obtain V(t^n+1,v) = V^n+1 and S^n+1 from the QR decomposition of L^n+1

# Inialize U, S, and V by taking the SVD of f(t_0, x, v) = f_0
from math import *
import numpy as np
import matplotlib.pyplot as plt

Rank = 5
nx, nv = (30, 30)
x = np.linspace(0, 1, nx)
v = np.linspace(0, 1, nv)
dx = (x[-1]-x[0])/(nx-1)
dv = (v[-1]-v[0])/(nv-1)
xm, vm = np.meshgrid(x, v, indexing='ij')
f_0 = (1 + 0.01*np.cos(2*xm))*np.cos(pi*vm - 1)

plt.figure()
plt.scatter(xm, vm)
cp = plt.pcolormesh(xm, vm, f_0, cmap='viridis')
plt.colorbar(cp)
plt.show()
...

# SVD of the IC f_0
U0, S0, Vh0 = np.linalg.svd(f_0, full_matrices=False)
S0 = np.diag(S0)
#
# plt.subplot(1,3,1)
# plt.imshow(U0)
# plt.subplot(1,3,2)
# plt.imshow(np.diag(S0))
# plt.subplot(1,3,3)
# plt.imshow(Vh0)
# plt.show()
#
# print(np.sqrt(np.sum((U@np.diag(S)@Vh - f_0.T)**2)))
...

# Finite difference matrices (should this be periodic?) #
D_x = np.zeros((nx, nx))
D_v = np.zeros((nv, nv))
for index in range(nx - 1):
    D_x[index, index:index + 2] = [-1/dx, 1/dx]
for index in range(nv - 1):
    D_v[index, index:index + 2] = [-1/dv, 1/dv]
D_x[-1][-1] = -1/dx
D_v[-1][-1] = -1/dv
# print(D_x)
# print(D_v)
...

# Equations (9)-(11)
def K_del_t(K_tx, Vh_tv, vel, E):
    K_del_x = D_x @ K_tx
    Vh_del_v = Vh_tv @ D_v.T

    proj_VvV = Vh_tv @ (vel * Vh_tv).T * dv
    proj_VdV = Vh_tv @ Vh_del_v.T * dv

    return -K_del_x @ proj_VvV + np.diag(E) @ K_tx @ proj_VdV

def S_del_t(U_tx, S_t, Vh_tv, vel, E):
    U_del_x = D_x @ U_tx
    Vh_del_v = Vh_tv @ D_v.T

    proj_UdU = U_tx.T @ U_del_x * dx
    proj_UEU = U_tx.T @ np.diag(E) @ U_tx * dx
    proj_VvV = Vh_tv @ (vel * Vh_tv).T * dv
    proj_VdV = Vh_tv @ Vh_del_v.T * dv

    return -proj_UdU @ S_t @ proj_VvV + proj_UEU @ S_t @ proj_VdV

def L_del_t(U_tx, L_tv, vel, E):
    U_del_x = D_x @ U_tx
    L_del_v = L_tv @ D_v.T

    proj_UdU = U_tx.T @ U_del_x * dx
    proj_UEU = U_tx.T @ np.diag(E) @ U_tx * dx

    return -proj_UdU @ L_tv * vel + proj_UEU @ L_del_v
    #                          ^-- Should this velocity term be here?

from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson
def Electric_Field(phase_density):
    rho = 1 - simpson(phase_density, x=v, axis=1)
    E = -cumulative_simpson(rho, x=x, initial=0)
    return E

U0_low = U0[:, :Rank]
S0_low = S0[:Rank, :Rank]
Vh0_low = Vh0[:Rank, :]
# K = U0_low @ S0_low #shape = (nx,rank)
# L = S0_low @ Vh0_low #shape = (rank,nv)
print(U0_low.shape, S0_low.shape, Vh0_low.shape)

# Fixed-stepsize RK4 method that works on matrices to avoid reshaping
def rk4(func, Y, dt):
    k1 = func(Y)
    k2 = func(Y + 0.5*dt*k1)
    k3 = func(Y + 0.5*dt*k2)
    k4 = func(Y + dt*k3)
    return Y + (k1 + 2*k2 + 2*k3 + k4)*dt/6

# Main loop for the KSL Procedure

# U, S, Vh = [U0_low], [S0_low], [Vh0_low]
U, S, Vh = U0_low, S0_low, Vh0_low
U_curr, S_curr, Vh_curr = U, S, Vh
T_span = [0, 1]
dT = 0.001
nt = int((T_span[1] - T_span[0])/dT)
for T in range(nt):
    # U_curr, S_curr, Vh_curr = U[-1], S[-1], Vh[-1]

    # K-Step
    def DtK(Kk):
        E = Electric_Field(Kk @ Vh_curr)
        return K_del_t(Kk, Vh_curr, v, E)
    K_next = rk4(DtK, U_curr @ S_curr, dT)
    U_next, S_star = np.linalg.qr(K_next)

    # S-Step
    def DtS(Sk):
        E = Electric_Field(U_next @ Sk @ Vh_curr)
        return -S_del_t(U_next, Sk, Vh_curr, v, E)
    S_star_star = rk4(DtS, S_star, -dT)

    # L-Step
    def DtL(Lk):
        E = Electric_Field(U_next @ Lk)
        return L_del_t(U_next, Lk, v, E)
    L_next = rk4(DtL, S_star_star @ Vh_curr, dT)
    Vh_next, S_next = np.linalg.qr(L_next.T)

    # Update current decomposition
    U_curr, S_curr, Vh_curr = U_next, S_next.T, Vh_next.T
    # U.append(U_next)
    # S.append(S_next)
    # Vh.append(Vh_next)

# Construct the final state of the system
# F = U[-1] @ S[-1] @ Vh[-1]
F = U_curr @ S_curr @ Vh_curr
plt.figure()
plt.scatter(xm, vm)
cp = plt.pcolormesh(xm, vm, F, cmap='viridis')
plt.colorbar(cp)
plt.show()