import numpy as np


nx = nv = 128
rank = 6
k = 10  # 控制基函数总数：cos、sin 交替，按频率递增。
alpha = 1e-3
beta = 0.2
v_bar = 2.4
T = 20
dx = 2 * np.pi / (beta * nx)
dv = 12 / nv
x = (np.arange(nx) + 0.5) * dx
v = -6 + (np.arange(nv) + 0.5) * dv
W_x = dx * np.eye(nx)
W_v = dv * np.eye(nv)

f_eq_v = (
    np.exp(-(v - v_bar) ** 2 / 2) + np.exp(-(v + v_bar) ** 2 / 2)
) / (2 * np.sqrt(2 * np.pi))
f_eq = np.ones((nx, 1)) * f_eq_v[None, :]

f_0 = (1 + alpha * np.cos(beta * x))[:, None] * f_eq_v[None, :]

if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
    raise ValueError("k must be an integer.")
if not 1 <= k <= 2 * ((nx - 1) // 2):
    raise ValueError(f"k must be between 1 and {2 * ((nx - 1) // 2)}.")

L_x = nx * dx
theta = 2 * np.pi * x / L_x
frequencies = np.arange(1, (k + 1) // 2 + 1)
H_basis = np.empty((k, nx))
H_basis[0::2] = np.cos(frequencies[:, None] * theta)
H_basis[1::2] = np.sin(frequencies[:k // 2, None] * theta)
# 零均值、加权正交归一：H_basis @ W_x @ H_basis.T = I。
H_basis *= np.sqrt(2 / L_x)

TOL = 1e-6
GRAD_TOL = 1e-6
MAX_STEPS = 20

def build_H(a_k):
    a_k = np.asarray(a_k, dtype=float)
    if a_k.shape != (k,):
        raise ValueError(f"a_k must have shape ({k},).")
    return a_k @ H_basis


def solve_E(U, S, V, *, return_multiplier=False):
    k_x = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    E_multiplier = np.zeros(nx, dtype=complex)
    E_multiplier[k_x != 0] = -1j / k_x[k_x != 0]
    if nx % 2 == 0:
        E_multiplier[nx // 2] = 0

    rho = 1.0 - dv * (U @ S @ np.sum(V, axis=0))
    E = np.fft.ifft(E_multiplier * np.fft.fft(rho)).real
    if return_multiplier:
        return E, E_multiplier  # Reuse the same multiplier in the adjoint.
    return E

def RHS(U, S, V, H):
    # Periodic central differences in x and v.
    f = U @ S @ V.T
    E = solve_E(U, S, V) + H
    df_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    df_dv = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dv)
    return -v[None, :] * df_dx + E[:, None] * df_dv

# Forward
class DLR:
    def __init__(self, U, S, V, dt, H):
        self.U = U.copy()
        self.S = S.copy()
        self.V = V.copy()
        self.dt = dt
        self.H = H.copy()
        self.U_history = [self.U.copy()]
        self.S_history = [self.S.copy()]
        self.V_history = [self.V.copy()]
        self.K_star_history = []
        self.S_hat_history = []
        self.S_tilde_history = []
        self.L_star_history = []

    def K_step(self):
        F_0 = RHS(self.U, self.S, self.V, self.H)
        K = self.U @ self.S
        self.K_star = K + self.dt * F_0 @ W_v @ self.V
        U_standard, S_standard = np.linalg.qr(self.K_star, mode="reduced")
        signs = np.where(np.diag(S_standard) < 0, -1.0, 1.0)
        U_standard = U_standard * signs[None, :]
        S_standard = signs[:, None] * S_standard
        self.U = U_standard / np.sqrt(dx)
        self.S_hat = np.sqrt(dx) * S_standard
        self.S = self.S_hat
        return self.U, self.S, self.V

    def S_step(self):
        F_1 = RHS(self.U, self.S, self.V, self.H)
        self.S_tilde = (
            self.S - self.dt * self.U.T @ W_x @ F_1 @ W_v @ self.V
        )
        self.S = self.S_tilde
        return self.U, self.S, self.V

    def L_step(self):
        F_2 = RHS(self.U, self.S, self.V, self.H)
        L = self.V @ self.S.T
        self.L_star = L + self.dt * F_2.T @ W_x @ self.U
        V_standard, S_T_standard = np.linalg.qr(self.L_star, mode="reduced")
        signs = np.where(np.diag(S_T_standard) < 0, -1.0, 1.0)
        V_standard = V_standard * signs[None, :]
        S_T_standard = signs[:, None] * S_T_standard
        self.V = V_standard / np.sqrt(dv)
        self.S = (np.sqrt(dv) * S_T_standard).T
        return self.U, self.S, self.V

    def step(self):
        self.K_step()
        self.S_step()
        self.L_step()
        self.U_history.append(self.U.copy())
        self.S_history.append(self.S.copy())
        self.V_history.append(self.V.copy())
        self.K_star_history.append(self.K_star.copy())
        self.S_hat_history.append(self.S_hat.copy())
        self.S_tilde_history.append(self.S_tilde.copy())
        self.L_star_history.append(self.L_star.copy())
        return self.U, self.S, self.V

    def run(self, num_steps):
        for _ in range(num_steps):
            self.step()
        return self.U, self.S, self.V


# Backward
class Adjoint:
    def __init__(self, forward):
        self.forward = forward
        self.dt = forward.dt
        self.H = forward.H.copy()
        self.N = len(forward.U_history) - 1
        self.U_bar_history = [None] * (self.N + 1)
        self.S_bar_history = [None] * (self.N + 1)
        self.V_bar_history = [None] * (self.N + 1)
        self.p_L_history = [None] * self.N
        self.p_S_history = [None] * self.N
        self.p_K_history = [None] * self.N
        self.p_L = None
        self.p_S = None
        self.p_K = None

    def initial_condition(self):
        U = self.forward.U_history[self.N]
        S = self.forward.S_history[self.N]
        V = self.forward.V_history[self.N]
        G = U @ S @ V.T - f_eq
        w = dx * dv
        U_bar = w * G @ V @ S.T
        S_bar = w * U.T @ G @ V
        V_bar = w * G.T @ U @ S
        return U_bar, S_bar, V_bar

    def DQ_adjoint(self, Q, R, B, w):
        # DQ(A)^*[B]
        M = B.T @ Q
        ltor = np.tril(M) + np.tril(M, -1).T
        A_bar = B - w * Q @ ltor
        return np.linalg.solve(R, A_bar.T).T

    def DR_adjoint(self, Q, R, C, w):
        # DR(A)^*[C]
        M = R @ C.T
        ltor = np.tril(M) + np.tril(M, -1).T
        A_bar = w * Q @ ltor
        return np.linalg.solve(R, A_bar.T).T

    def RHS(self, U, S, V):
        return RHS(U, S, V, self.H)

    def Df_RHS_adjoint(self, U, S, V, F_bar):
        f = U @ S @ V.T
        E, E_multiplier = solve_E(U, S, V, return_multiplier=True)
        E = E + self.H
        dF_bar_dx = (
            np.roll(F_bar, -1, axis=0) - np.roll(F_bar, 1, axis=0)
        ) / (2 * dx)
        dF_bar_dv = (
            np.roll(F_bar, -1, axis=1) - np.roll(F_bar, 1, axis=1)
        ) / (2 * dv)
        g = v[None, :] * dF_bar_dx - E[:, None] * dF_bar_dv

        df_dv = (
            np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
        ) / (2 * dv)
        E_bar = np.sum(F_bar * df_dv, axis=1)
        rho_bar = np.fft.ifft(
            np.conj(E_multiplier) * np.fft.fft(E_bar)
        ).real
        g = g - dv * rho_bar[:, None]
        return g

    def solve_p_L(self):
        p_L = self.DQ_adjoint(
            self.V_next, self.S_next.T, self.V_bar_next, dv
        ) + self.DR_adjoint(
            self.V_next, self.S_next.T, self.S_bar_next.T, dv
        )
        return p_L

    def solve_p_S(self):
        self.F_2_bar = self.dt * W_x @ self.U_next @ self.p_L.T
        self.g_2 = self.Df_RHS_adjoint(
            self.U_next, self.S_tilde, self.V, self.F_2_bar
        )
        p_S = self.p_L.T @ self.V + self.U_next.T @ self.g_2 @ self.V
        return p_S

    def solve_p_K(self):
        self.F_1 = self.RHS(self.U_next, self.S_hat, self.V)
        self.F_2 = self.RHS(self.U_next, self.S_tilde, self.V)
        self.F_1_bar = (
            -self.dt * W_x @ self.U_next @ self.p_S @ self.V.T @ W_v
        )
        self.g_1 = self.Df_RHS_adjoint(
            self.U_next, self.S_hat, self.V, self.F_1_bar
        )

        self.C_U = (
            self.U_bar_next
            - self.dt * W_x @ self.F_1 @ W_v @ self.V @ self.p_S.T
            + self.g_1 @ self.V @ self.S_hat.T
            + self.dt * W_x @ self.F_2 @ self.p_L
            + self.g_2 @ self.V @ self.S_tilde.T
        )
        self.C_S_hat = self.p_S + self.U_next.T @ self.g_1 @ self.V

        p_K = self.DQ_adjoint(
            self.U_next, self.S_hat, self.C_U, dx
        ) + self.DR_adjoint(
            self.U_next, self.S_hat, self.C_S_hat, dx
        )
        return p_K

    def adjoint_update(self):
        self.F_0 = self.RHS(self.U, self.S, self.V)
        self.F_0_bar = self.dt * self.p_K @ self.V.T @ W_v
        self.g_0 = self.Df_RHS_adjoint(
            self.U, self.S, self.V, self.F_0_bar
        )
        U_bar = self.p_K @ self.S.T + self.g_0 @ self.V @ self.S.T
        S_bar = self.U.T @ self.p_K + self.U.T @ self.g_0 @ self.V
        V_bar = (
            self.dt * W_v @ self.F_0.T @ self.p_K
            + self.g_0.T @ self.U @ self.S
            - self.dt * W_v @ self.F_1.T @ W_x @ self.U_next @ self.p_S
            + self.g_1.T @ self.U_next @ self.S_hat
            + self.p_L @ self.S_tilde
            + self.g_2.T @ self.U_next @ self.S_tilde
        )
        return U_bar, S_bar, V_bar

    def step(self, n):
        self.n = n
        self.U = self.forward.U_history[n]
        self.S = self.forward.S_history[n]
        self.V = self.forward.V_history[n]
        self.U_next = self.forward.U_history[n + 1]
        self.S_next = self.forward.S_history[n + 1]
        self.V_next = self.forward.V_history[n + 1]
        self.K_star = self.forward.K_star_history[n]
        self.S_hat = self.forward.S_hat_history[n]
        self.S_tilde = self.forward.S_tilde_history[n]
        self.L_star = self.forward.L_star_history[n]
        self.U_bar_next = self.U_bar_history[n + 1].copy()
        self.S_bar_next = self.S_bar_history[n + 1].copy()
        self.V_bar_next = self.V_bar_history[n + 1].copy()
        self.p_L = self.solve_p_L()
        self.p_S = self.solve_p_S()
        self.p_K = self.solve_p_K()
        self.U_bar, self.S_bar, self.V_bar = self.adjoint_update()
        self.p_L_history[n] = self.p_L.copy()
        self.p_S_history[n] = self.p_S.copy()
        self.p_K_history[n] = self.p_K.copy()
        self.U_bar_history[n] = self.U_bar.copy()
        self.S_bar_history[n] = self.S_bar.copy()
        self.V_bar_history[n] = self.V_bar.copy()
        return self.U_bar, self.S_bar, self.V_bar

    def run(self):
        (
            self.U_bar_history[self.N],
            self.S_bar_history[self.N],
            self.V_bar_history[self.N],
        ) = self.initial_condition()

        for n in range(self.N - 1, -1, -1):
            self.step(n)

        return {
            "p_L": self.p_L_history,
            "p_S": self.p_S_history,
            "p_K": self.p_K_history,
            "U_bar": self.U_bar_history,
            "S_bar": self.S_bar_history,
            "V_bar": self.V_bar_history,
        }


def DH_RHS_adjoint(U, S, V, H, F_bar):
    # calculate f_a = U @ S @ V.T at DH_RHS(f_a, H)^*[F_bar]
    f = U @ S @ V.T
    df_dv = (
        np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
    ) / (2 * dv)
    return np.sum(F_bar * df_dv, axis=1)


def gradient_H(forward, adjoint_result):
    N = len(forward.K_star_history)
    dt = forward.dt
    J_H_steps = np.zeros((N, nx))

    for n in range(N):
        U = forward.U_history[n]
        S = forward.S_history[n]
        V = forward.V_history[n]
        U_next = forward.U_history[n + 1]
        S_hat = forward.S_hat_history[n]
        S_tilde = forward.S_tilde_history[n]
        p_K = adjoint_result["p_K"][n]
        p_S = adjoint_result["p_S"][n]
        p_L = adjoint_result["p_L"][n]

        F_0_bar = dt * p_K @ V.T @ W_v
        F_1_bar = -dt * W_x @ U_next @ p_S @ V.T @ W_v
        F_2_bar = dt * W_x @ U_next @ p_L.T

        J_H_steps[n] = (
            DH_RHS_adjoint(U, S, V, forward.H, F_0_bar)
            + DH_RHS_adjoint(U_next, S_hat, V, forward.H, F_1_bar)
            + DH_RHS_adjoint(U_next, S_tilde, V, forward.H, F_2_bar)
        )

    return np.sum(J_H_steps, axis=0)


def gradient_a_k(J_H):
    J_H = np.asarray(J_H, dtype=float)
    if J_H.shape != (nx,):
        raise ValueError(f"J_H must have shape ({nx},).")
    # J_H 已是对网格值 H 的梯度；链式法则无需再乘 W_x。
    return H_basis @ J_H


def update_a_k(a_k, J_a, h):
    a_k = np.asarray(a_k, dtype=float)
    J_a = np.asarray(J_a, dtype=float)
    if a_k.shape != (k,) or J_a.shape != (k,):
        raise ValueError(f"a_k and J_a must both have shape ({k},).")
    a_k_next = a_k - h * J_a
    H_next = build_H(a_k_next)
    return a_k_next, H_next


def initial_factors():
    U_svd, singular_values, Vt_svd = np.linalg.svd(f_0, full_matrices=False)
    U_0 = U_svd[:, :rank] / np.sqrt(dx)
    V_0 = Vt_svd[:rank].T / np.sqrt(dv)
    S_0 = np.sqrt(dx * dv) * np.diag(singular_values[:rank])
    return U_0, S_0, V_0


def objective(forward):
    # 与 Adjoint.initial_condition 的终端目标一致
    f_N = forward.U @ forward.S @ forward.V.T
    return 0.5 * dx * dv * np.sum((f_N - f_eq) ** 2)




def solve_control(a_k, U_0, S_0, V_0, dt, T):
    N = int(round(T / dt))

    forward = DLR(U_0, S_0, V_0, dt, build_H(a_k))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        forward.run(N)
        J = objective(forward)
    if not np.isfinite(J):
        raise FloatingPointError("Non-finite forward objective.")
    return forward, J




# Line search
def optimize(a_k, dt=0.005, T=T, h=0.01, tol=TOL, grad_tol=GRAD_TOL, max_steps=MAX_STEPS):
    a_k = np.array(a_k, dtype=float, copy=True)
    U_0, S_0, V_0 = initial_factors()
    forward, J = solve_control(a_k, U_0, S_0, V_0, dt, T)
    J_history = [J]
    grad_history = []
    h_history = []
    stop_reason = "Max steps reached"

    for step in range(max_steps):
        if J <= tol:
            stop_reason = "Objective tolerance reached"
            break

        # Compute gradient
        adjoint_result = Adjoint(forward).run()
        J_H = gradient_H(forward, adjoint_result)
        J_a = gradient_a_k(J_H)
        del adjoint_result
        grad_norm = np.linalg.norm(J_a)
        grad_history.append(grad_norm)
        print(f"Iter {step}: J={J:.8e}, |g|={grad_norm:.3e}",
              flush=True)


        if not np.isfinite(grad_norm):
            stop_reason = "Non-finite gradient"
            break
        if grad_norm <= grad_tol:
            stop_reason = "Gradient tolerance reached"
            break

        h_used = max(2 * h_history[-1], h) if h_history else h
        accepted = False
        for _ in range(30):
            a_trial, _ = update_a_k(a_k, J_a, h_used)
            try:
                trial, J_trial = solve_control(a_trial, U_0, S_0, V_0, dt, T)
            except (FloatingPointError, np.linalg.LinAlgError):
                h_used *= 0.5
                continue
            if J_trial < J and J_trial <= J - 0.1 * h_used * grad_norm**2:
                accepted = True
                break
            h_used *= 0.5

        if not accepted:
            stop_reason = "Line search failed (30 trials)"
            break

        a_k, forward, J = a_trial, trial, J_trial
        J_history.append(J)
        h_history.append(h_used)
        print(f"Accepted {step + 1}: J={J:.8e}, h={h_used:.3e}",
              flush=True)

    if J <= tol:
        stop_reason = "Objective tolerance reached"

    return {
        "a_k": a_k,
        "H": forward.H.copy(),
        "forward": forward,
        "J_history": np.array(J_history),
        "grad_history": np.array(grad_history),
        "h_history": np.array(h_history),
        "stop_reason": stop_reason,
    }



#plot finished by codex
def plot_result(forward, times=(0, T/2, T), filename=None):
    from pathlib import Path
    import matplotlib.pyplot as plt
    times = np.asarray(times, dtype=float)
    T = (len(forward.U_history) - 1) * forward.dt
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty, finite 1D array.")
    if np.any(times < 0) or np.any(times > T):
        raise ValueError("Plot times must lie within [0, T].")
    indices = np.rint(times / forward.dt).astype(int)
    f_plot = [f_eq]
    titles = [r"$f^{eq}$"]
    for n in indices:
        f_n = (forward.U_history[n] @ forward.S_history[n]
               @ forward.V_history[n].T)
        f_plot.append(f_n)
        titles.append(f"T = {n * forward.dt:g}")

    vmin = min(np.min(f_n) for f_n in f_plot)
    vmax = max(np.max(f_n) for f_n in f_plot)
    fig, axes = plt.subplots(1, len(f_plot), figsize=(4 * len(f_plot), 3.6),
                             constrained_layout=True)
    for ax, f_n, title in zip(axes, f_plot, titles):
        image = ax.pcolormesh(x, v, f_n.T, shading="auto", cmap="viridis",
                              vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("v")
    fig.colorbar(image, ax=axes, label="f(x, v)")
    if filename is None:
        filename = Path(__file__).resolve().parent / "figures" / "Adjoint_DLR_central_difference_distribution.png"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    return fig, axes




def main(dt=0.005, T=T, h=0.1, tol=TOL, grad_tol=GRAD_TOL, max_steps=MAX_STEPS):
    from pathlib import Path

    # 不同 k 使用同一个非零初始控制，新增基函数的系数从零开始。
    a_k = np.zeros(k)
    a_k[0] = 1e-3
    result = optimize(a_k, dt, T, h, tol, grad_tol, max_steps)
    output_dir = Path(__file__).resolve().parent
    # Save the result and some snapshots of the distribution function at t=0, T/2, T
    forward = result["forward"]
    indices = np.rint(np.array([0, T / 2, T]) / dt).astype(int)
    snapshots = np.array([forward.U_history[n] @ forward.S_history[n]
                          @ forward.V_history[n].T for n in indices])
    np.savez(output_dir / "Adjoint_DLR_central_difference_result.npz",
             a_k=result["a_k"], H=result["H"],
             k=k, H_basis=H_basis, control_basis="normalized_fourier_cos_sin_zero_mean",
             J_history=result["J_history"], grad_history=result["grad_history"],
             h_history=result["h_history"], stop_reason=result["stop_reason"],
             dt=dt, T=T, tol=tol, grad_tol=grad_tol, max_steps=max_steps,
             spatial_scheme="central_difference", time_scheme="Euler_KSL",
             boundary_x="periodic", boundary_v="periodic",
             rank=forward.S.shape[0], electric_field_solver="Fourier",
             x=x, v=v, times=indices * dt, f=snapshots, f_eq=f_eq)
    print(f"Stop: {result['stop_reason']} (steps={len(result['h_history'])})")
    print(f"Final J={result['J_history'][-1]:.8e}")
    print(f"a_k = {result['a_k']}")
    plot_file = output_dir / "figures" / "Adjoint_DLR_central_difference_distribution.png"
    plot_result(forward, times=(0, T / 2, T), filename=plot_file)
    print(f"Plot: {plot_file}")
    return result


if __name__ == "__main__":
    result = main()

'''
(base) wrc@MacBookAir low-rank % python -u "/Users/wrc/Wisc/low-rank/Adjoint for DLR_central difference.py"
Iter 0: J=5.69456520e-02, |g|=6.991e+00
Accepted 1: J=5.20640308e-02, h=1.953e-04
Iter 1: J=5.20640308e-02, |g|=4.858e+00
Accepted 2: J=2.85460247e-02, h=3.125e-03
Iter 2: J=2.85460247e-02, |g|=3.631e+01
Accepted 3: J=3.73879603e-05, h=1.953e-04
Iter 3: J=3.73879603e-05, |g|=7.091e+02
Stop: Line search failed (30 trials) (steps=3)
Final J=3.73879603e-05
a_k = [ 1.62195906e-03  2.85474978e-04  1.93560260e-04  1.86649366e-04
  1.67200301e-03  3.43744933e-05 -2.05628722e-03  2.33575535e-04
 -9.52009837e-04  7.27741968e-05]
Plot: /Users/wrc/Wisc/low-rank/figures/Adjoint_DLR_central_difference_distribution.png
(base) wrc@MacBookAir low-rank % python -u "/Users/wrc/Wisc/low-rank/Adjoint for DLR_central difference.py"
Iter 0: J=6.07016051e-02, |g|=9.863e+00
Accepted 1: J=5.39534099e-02, h=1.953e-04
Iter 1: J=5.39534099e-02, |g|=4.982e+00
Accepted 2: J=7.34303007e-03, h=3.125e-03
Iter 2: J=7.34303007e-03, |g|=1.639e+01
Accepted 3: J=6.51843000e-03, h=6.104e-06
Iter 3: J=6.51843000e-03, |g|=5.746e+00
Accepted 4: J=1.84123008e-04, h=1.953e-04
Iter 4: J=1.84123008e-04, |g|=2.989e+02
Accepted 5: J=1.62998391e-04, h=7.451e-10
Iter 5: J=1.62998391e-04, |g|=7.064e+01
Accepted 6: J=5.37601463e-05, h=1.907e-07
Iter 6: J=5.37601463e-05, |g|=7.948e+01
Accepted 7: J=3.49528332e-05, h=1.192e-08
Iter 7: J=3.49528332e-05, |g|=3.167e+01
Accepted 8: J=3.24479740e-05, h=1.192e-08
Iter 8: J=3.24479740e-05, |g|=1.317e+01
Accepted 9: J=3.15944906e-05, h=1.192e-08
Iter 9: J=3.15944906e-05, |g|=3.287e+00
Accepted 10: J=3.15175867e-05, h=2.384e-08
Iter 10: J=3.15175867e-05, |g|=4.038e+00
Accepted 11: J=3.14639338e-05, h=2.384e-08
Iter 11: J=3.14639338e-05, |g|=5.116e+00
Accepted 12: J=3.13051821e-05, h=1.192e-08
Iter 12: J=3.13051821e-05, |g|=2.254e+00
Accepted 13: J=3.11449557e-05, h=9.537e-08
Iter 13: J=3.11449557e-05, |g|=7.514e+00
Accepted 14: J=3.08442533e-05, h=1.192e-08
Iter 14: J=3.08442533e-05, |g|=2.533e+00
Accepted 15: J=3.07804497e-05, h=4.768e-08
Iter 15: J=3.07804497e-05, |g|=5.945e+00
Accepted 16: J=3.05848416e-05, h=1.192e-08
Iter 16: J=3.05848416e-05, |g|=2.369e+00
Accepted 17: J=3.04754463e-05, h=4.768e-08
Iter 17: J=3.04754463e-05, |g|=4.927e+00
Accepted 18: J=3.03325213e-05, h=1.192e-08
Iter 18: J=3.03325213e-05, |g|=2.232e+00
Accepted 19: J=3.02642235e-05, h=9.537e-08
Iter 19: J=3.02642235e-05, |g|=8.448e+00
Accepted 20: J=2.99129524e-05, h=1.192e-08
Stop: Max steps reached (steps=20)
Final J=2.99129524e-05
a_k = [ 2.21373962e-03  3.16047739e-04 -1.32574961e-03  3.54580051e-04
  1.99674993e-03  1.38245280e-04 -1.12258365e-03  2.53776112e-04
 -4.99290995e-04  9.74531777e-05]
Plot: /Users/wrc/Wisc/low-rank/figures/Adjoint_DLR_central_difference_distribution.png
(base) wrc@MacBookAir low-rank % python -u "/Users/wrc/Wisc/low-rank/Adjoint for DLR.py"
Iter 0: J=3.92755693e-03, |g|=1.600e+00
Accepted 1: J=4.60078833e-06, h=3.125e-03
Iter 1: J=4.60078833e-06, |g|=2.039e-02
Accepted 2: J=4.09445295e-06, h=6.250e-03
Iter 2: J=4.09445295e-06, |g|=1.539e-02
Accepted 3: J=3.73082288e-06, h=3.125e-03
Iter 3: J=3.73082288e-06, |g|=8.144e-04
Accepted 4: J=3.72866464e-06, h=6.250e-03
Iter 4: J=3.72866464e-06, |g|=3.589e-04
Accepted 5: J=3.72839352e-06, h=1.250e-02
Iter 5: J=3.72839352e-06, |g|=7.449e-04
Accepted 6: J=3.72789826e-06, h=6.250e-03
Iter 6: J=3.72789826e-06, |g|=7.391e-04
Accepted 7: J=3.72740212e-06, h=6.250e-03
Iter 7: J=3.72740212e-06, |g|=7.339e-04
Accepted 8: J=3.72689989e-06, h=6.250e-03
Iter 8: J=3.72689989e-06, |g|=7.289e-04
Accepted 9: J=3.72639459e-06, h=6.250e-03
Iter 9: J=3.72639459e-06, |g|=7.241e-04
Accepted 10: J=3.72588314e-06, h=6.250e-03
Iter 10: J=3.72588314e-06, |g|=7.198e-04
Accepted 11: J=3.72536923e-06, h=6.250e-03
Iter 11: J=3.72536923e-06, |g|=7.149e-04
Accepted 12: J=3.72484472e-06, h=6.250e-03
Iter 12: J=3.72484472e-06, |g|=7.098e-04
Accepted 13: J=3.72431918e-06, h=6.250e-03
Iter 13: J=3.72431918e-06, |g|=7.052e-04
Accepted 14: J=3.72378530e-06, h=6.250e-03
Iter 14: J=3.72378530e-06, |g|=7.008e-04
Accepted 15: J=3.72324985e-06, h=6.250e-03
Iter 15: J=3.72324985e-06, |g|=6.969e-04
Accepted 16: J=3.72270721e-06, h=6.250e-03
Iter 16: J=3.72270721e-06, |g|=6.932e-04
Accepted 17: J=3.72216085e-06, h=6.250e-03
Iter 17: J=3.72216085e-06, |g|=6.896e-04
Accepted 18: J=3.72160673e-06, h=6.250e-03
Iter 18: J=3.72160673e-06, |g|=6.863e-04
Accepted 19: J=3.72104985e-06, h=6.250e-03
Iter 19: J=3.72104985e-06, |g|=6.831e-04
Accepted 20: J=3.72048366e-06, h=6.250e-03
Stop: Max steps reached (steps=20)
Final J=3.72048366e-06
a_k = [ 4.81271165e-05  4.82141897e-03 -2.20110379e-04 -9.43713553e-05
 -9.09894989e-05  2.53811226e-04 -5.86449595e-05 -1.78103455e-05
 -5.61127789e-05  9.73865728e-05]
Plot: /Users/wrc/Wisc/low-rank/figures/Adjoint_DLR_distribution.png
(base) wrc@MacBookAir low-rank % python -u "/Users/wrc/Wisc/low-rank/Adjoint for DLR_central difference.py"
Iter 0: J=4.95589670e-03, |g|=1.936e+00
Accepted 1: J=1.90955681e-04, h=3.125e-03
Iter 1: J=1.90955681e-04, |g|=3.810e-01
Accepted 2: J=1.45120203e-05, h=3.125e-03
Iter 2: J=1.45120203e-05, |g|=8.717e-02
Accepted 3: J=5.42646493e-06, h=3.125e-03
Iter 3: J=5.42646493e-06, |g|=2.097e-02
Accepted 4: J=4.91576857e-06, h=3.125e-03
Iter 4: J=4.91576857e-06, |g|=5.375e-03
Accepted 5: J=4.88188571e-06, h=3.125e-03
Iter 5: J=4.88188571e-06, |g|=1.474e-03
'''