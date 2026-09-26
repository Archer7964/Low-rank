import numpy as np
from scipy.optimize import minimize


nx = nv = 128

ranks = [14, 15, 24]

k = 10
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
H_basis *= np.sqrt(2 / L_x)

TOL = 1e-15
GRAD_TOL = 1e-15
MAX_STEPS = 200
LBFGS_MAXCOR = 10
LBFGS_FTOL = 1e-14
LBFGS_MAXLS = 30

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
        return E, E_multiplier
    return E


def RHS(U, S, V, H):
    f = U @ S @ V.T
    E = solve_E(U, S, V) + H
    df_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    df_dv = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dv)
    d2f_dx2 = (
        np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)
    ) / dx**2
    d2f_dv2 = (
        np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
    ) / dv**2
    return (
        -v[None, :] * df_dx + E[:, None] * df_dv
        + 0.5 * dx * np.abs(v)[None, :] * d2f_dx2
        + 0.5 * dv * np.abs(E)[:, None] * d2f_dv2
    )


# Forward
class DLR:
    def __init__(self, U, S, V, dt, H):
        self.U = U.copy()
        self.S = S.copy()
        self.V = V.copy()
        self.dt = dt
        self.H = H.copy()
        self.rank = S.shape[0]
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
        M = B.T @ Q
        ltor = np.tril(M) + np.tril(M, -1).T
        A_bar = B - w * Q @ ltor
        return np.linalg.solve(R, A_bar.T).T

    def DR_adjoint(self, Q, R, C, w):
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
        d2F_bar_dx2 = (
            np.roll(F_bar, -1, axis=0) - 2 * F_bar
            + np.roll(F_bar, 1, axis=0)
        ) / dx**2
        d2F_bar_dv2 = (
            np.roll(F_bar, -1, axis=1) - 2 * F_bar
            + np.roll(F_bar, 1, axis=1)
        ) / dv**2
        g = (
            v[None, :] * dF_bar_dx - E[:, None] * dF_bar_dv
            + 0.5 * dx * np.abs(v)[None, :] * d2F_bar_dx2
            + 0.5 * dv * np.abs(E)[:, None] * d2F_bar_dv2
        )

        df_dv = (
            np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
        ) / (2 * dv)
        d2f_dv2 = (
            np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
        ) / dv**2
        phi = df_dv + 0.5 * dv * np.sign(E)[:, None] * d2f_dv2
        E_bar = np.sum(F_bar * phi, axis=1)
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
        self.F_2_bar = self.dt * W_x @ self.U_next @ self.p_L.T #type: ignore
        self.g_2 = self.Df_RHS_adjoint(
            self.U_next, self.S_tilde, self.V, self.F_2_bar
        )
        p_S = self.p_L.T @ self.V + self.U_next.T @ self.g_2 @ self.V   #type: ignore
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
            - self.dt * W_x @ self.F_1 @ W_v @ self.V @ self.p_S.T  #type: ignore
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
    # 在 f_a = U @ S @ V.T 处计算 DH_RHS(f_a, H)^*[F_bar]
    f = U @ S @ V.T
    E = solve_E(U, S, V) + H
    df_dv = (
        np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
    ) / (2 * dv)
    d2f_dv2 = (
        np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
    ) / dv**2
    phi = df_dv + 0.5 * dv * np.sign(E)[:, None] * d2f_dv2
    return np.sum(F_bar * phi, axis=1)


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
    return H_basis @ J_H


def update_a_k(a_k, J_a, h):
    a_k = np.asarray(a_k, dtype=float)
    J_a = np.asarray(J_a, dtype=float)
    if a_k.shape != (k,) or J_a.shape != (k,):
        raise ValueError(f"a_k and J_a must both have shape ({k},).")
    a_k_next = a_k - h * J_a
    H_next = build_H(a_k_next)
    return a_k_next, H_next


def validate_rank(rank):
    if (not isinstance(rank, (int, np.integer))
            or isinstance(rank, (bool, np.bool_)) or not 1 <= rank <= min(nx, nv)):
        raise ValueError(f"rank must be an integer in [1, {min(nx, nv)}].")


def initial_factors(rank):
    validate_rank(rank)
    U_svd, singular_values, Vt_svd = np.linalg.svd(f_0, full_matrices=False)
    # 保持 U.T @ W_x @ U = I、V.T @ W_v @ V = I。
    U_0 = U_svd[:, :rank] / np.sqrt(dx)
    V_0 = Vt_svd[:rank].T / np.sqrt(dv)
    S_0 = np.sqrt(dx * dv) * np.diag(singular_values[:rank])
    return U_0, S_0, V_0


def objective(forward):
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


# L-BFGS: use the existing KSL forward solve and adjoint gradient.
def optimize(a_k, dt=0.005, T=T, tol=TOL, grad_tol=GRAD_TOL,
             max_steps=MAX_STEPS, *, rank, log=print):
    a_k = np.array(a_k, dtype=float, copy=True)
    U_0, S_0, V_0 = initial_factors(rank)
    forward, J = solve_control(a_k, U_0, S_0, V_0, dt, T)
    J_history = [J]
    grad_history = []
    step_norm_history = []
    previous_a = a_k.copy()
    latest_gradient = None
    steps = 0
    stop_reason = "Max steps reached"
    log(f"[rank={rank}] Initial: J_model={J:.10e}")

    def value_and_gradient(a):
        nonlocal latest_gradient
        trajectory, value = solve_control(a, U_0, S_0, V_0, dt, T)
        adjoint_result = Adjoint(trajectory).run()
        J_H = gradient_H(trajectory, adjoint_result)
        latest_gradient = gradient_a_k(J_H)
        return value, latest_gradient

    def record_iteration(intermediate_result):
        nonlocal previous_a
        # Callback runs after an accepted iteration; the latest gradient is at this point.
        J_history.append(intermediate_result.fun)
        grad_norm = np.linalg.norm(latest_gradient, ord=np.inf)
        step_norm = np.linalg.norm(intermediate_result.x - previous_a)
        grad_history.append(grad_norm)
        step_norm_history.append(step_norm)
        previous_a = intermediate_result.x.copy()
        log(f"[rank={rank}] Iter {len(step_norm_history)}: "
            f"J_model={intermediate_result.fun:.10e}, |g|_inf={grad_norm:.6e}, "
            f"|delta_a|={step_norm:.6e}")
        if intermediate_result.fun <= tol:
            raise StopIteration

    if J > tol and max_steps > 0:
        del forward
        result = minimize(
            value_and_gradient, a_k, method="L-BFGS-B", jac=True,
            callback=record_iteration,
            options={"maxiter": max_steps, "maxcor": LBFGS_MAXCOR,
                     "gtol": grad_tol, "ftol": LBFGS_FTOL, "maxls": LBFGS_MAXLS},
        )
        a_k = result.x.copy()
        steps = result.nit
        stop_reason = str(result.message)
        # Recompute at the returned coefficients so the saved state and final J agree.
        forward, J = solve_control(a_k, U_0, S_0, V_0, dt, T)
        J_history[-1] = J

    if J <= tol:
        stop_reason = "Objective tolerance reached"
    log(f"[rank={rank}] Stop: {stop_reason} (steps={steps})")

    return {
        "a_k": a_k,
        "H": forward.H.copy(),
        "forward": forward,
        "J_history": np.array(J_history),
        "grad_history": np.array(grad_history),
        "step_norm_history": np.array(step_norm_history),
        "steps": steps,
        "stop_reason": stop_reason,
    }


def plot_result(forward, times=(0, T/2, T), filename=None):
    from pathlib import Path
    import matplotlib.pyplot as plt
    times = np.asarray(times, dtype=float)
    T = (len(forward.U_history) - 1) * forward.dt
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty, finite 1D array.")
    if np.any(times < 0) or np.any(times > T + 1e-12):
        raise ValueError("Plot times must lie within [0, T].")
    indices = np.rint(times / forward.dt).astype(int)
    delta_f_plot = []
    titles = []
    for n in indices:
        f_n = (forward.U_history[n] @ forward.S_history[n]
               @ forward.V_history[n].T)
        delta_f_plot.append(f_n - f_eq)
        titles.append(f"T = {n * forward.dt:g}")

    limit = max(max(np.max(np.abs(delta_f)) for delta_f in delta_f_plot), 1e-15)
    fig, axes = plt.subplots(1, len(delta_f_plot), figsize=(4 * len(delta_f_plot), 3.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, delta_f, title in zip(axes, delta_f_plot, titles):
        image = ax.pcolormesh(x, v, delta_f.T, shading="auto", cmap="RdBu_r",
                              vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("v")
    fig.suptitle(f"rank={forward.rank}, k={k}, dt={forward.dt:g}")
    fig.colorbar(image, ax=axes, label=r"$f(x,v)-f^{\mathrm{eq}}(v)$")
    if filename is None:
        filename = Path(__file__).resolve().parent / "figures" / f"Adjoint_DLR_rank_{forward.rank}_distribution.png"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180) #type: ignore
    plt.close(fig)
    return fig, axes


def main(dt=0.005, T=T, tol=TOL, grad_tol=GRAD_TOL, max_steps=MAX_STEPS):
    import csv
    from datetime import datetime
    from pathlib import Path
    import matplotlib.pyplot as plt

    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be nonempty and contain no duplicates.")
    for rank in ranks:
        validate_rank(rank)
    output_dir = Path(__file__).resolve().parent
    csv_file = output_dir / "Adjoint_DLR_rank_results.csv"
    plot_file = output_dir / "figures" / "Adjoint_DLR_rank_comparison.png"
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    run_started = datetime.now().isoformat(timespec="seconds")
    a_initial = np.zeros(k)
    a_initial[0] = 1e-3
    reference_rank = min(nx, nv)
    U_ref, S_ref, V_ref = initial_factors(reference_rank)
    rows = []
    print(f"ranks={ranks}, k={k}, grid={nx}x{nv}, T={T:g}, dt={dt:g}, "
          f"max_steps={max_steps}", flush=True)
    print("J_model: current rank; J_ref: same control in full-rank KSL.", flush=True)

    for rank in sorted(ranks):
        with (log_dir / f"rank_{rank}.log").open("a", encoding="utf-8") as log_file:
            def log(message):
                print(message, flush=True)
                print(message, file=log_file, flush=True)

            log(f"\n--- {run_started} | rank={rank}, k={k}, grid={nx}x{nv}, "
                f"T={T:g}, dt={dt:g}, method=L-BFGS-B, max_steps={max_steps}, "
                f"tol={tol:g}, gtol={grad_tol:g}, ftol={LBFGS_FTOL:g}, "
                f"maxcor={LBFGS_MAXCOR}, maxls={LBFGS_MAXLS} ---")
            result = None
            J_model = J_ref = np.nan
            a_final = np.full(k, np.nan)
            steps = 0
            try:
                result = optimize(a_initial, dt=dt, T=T, tol=tol,
                                  grad_tol=grad_tol, max_steps=max_steps, rank=rank, log=log)
                J_model = objective(result["forward"])
                result["J_history"][-1] = J_model
                a_final = result["a_k"]
                steps = result["steps"]
                stop = result["stop_reason"]
                if rank == reference_rank:
                    J_ref = J_model
                else:
                    reference, J_ref = solve_control(a_final, U_ref, S_ref, V_ref, dt, T)
                    del reference
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                stop = f"{'Reference' if result is not None else 'Forward'} failed: {error}"
            row = dict(rank=rank, k=k, T=T, dt=dt, steps=steps,
                       J_model=J_model, J_ref=J_ref, stop=stop)
            row.update({f"a_{i + 1}": value for i, value in enumerate(a_final)})
            rows.append(row)
            log(f"rank={rank:3d} | steps={steps:2d} | J_model={J_model:.10e} | "
                f"J_ref={J_ref:.10e} | {stop}")
        with csv_file.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        del result

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.plot([row["rank"] for row in rows], [row["J_ref"] for row in rows], "o-")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(ranks), [str(rank) for rank in sorted(ranks)])
    positive_J = [row["J_ref"] for row in rows
                  if np.isfinite(row["J_ref"]) and row["J_ref"] > 0]
    if positive_J:
        ax.set_yscale("log")
        ax.set_ylim(min(positive_J) / 2, max(positive_J) * 2)
    ax.set_xlabel("rank")
    ax.set_ylabel("J_ref (full-rank KSL evaluation)")
    ax.set_title(f"L-BFGS-B | k={k}, T={T:g}, dt={dt:g}, max_steps={max_steps}")
    ax.grid(True, alpha=0.3)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=180) #type: ignore
    plt.close(fig)
    print(f"CSV: {csv_file}\nPlot: {plot_file}\nLogs: {log_dir}")
    return rows


if __name__ == "__main__":
    result = main()
