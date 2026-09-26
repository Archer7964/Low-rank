"""
Full-matrix reference for the original square-grid, full-rank Euler KSL scheme.

Forward and adjoint operate on f directly: no low-rank factors or QR derivatives.
This is a reference for the same discrete scheme, not an exact continuum solution.
"""

import numpy as np
from scipy.optimize import minimize


nx = nv = 128

k = 10
alpha = 1e-3
beta = 0.2
v_bar = 2.4
T = 20
dx = 2 * np.pi / (beta * nx)
dv = 12 / nv
x = (np.arange(nx) + 0.5) * dx
v = -6 + (np.arange(nv) + 0.5) * dv

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


def solve_E(f, *, return_multiplier=False):
    k_x = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    E_multiplier = np.zeros(nx, dtype=complex)
    E_multiplier[k_x != 0] = -1j / k_x[k_x != 0]
    if nx % 2 == 0:
        E_multiplier[nx // 2] = 0

    rho = 1.0 - dv * np.sum(f, axis=1)
    E = np.fft.ifft(E_multiplier * np.fft.fft(rho)).real
    if return_multiplier:
        return E, E_multiplier
    return E


def RHS(f, H):
    E = solve_E(f) + H
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


def RHS_adjoint(f, H, F_bar):
    """Return D_f RHS^*[F_bar] and D_H RHS^*[F_bar] in Frobenius coordinates."""
    E, E_multiplier = solve_E(f, return_multiplier=True)
    E = E + H
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
    f_bar = (
        v[None, :] * dF_bar_dx - E[:, None] * dF_bar_dv
        + 0.5 * dx * np.abs(v)[None, :] * d2F_bar_dx2
        + 0.5 * dv * np.abs(E)[:, None] * d2F_bar_dv2
    )

    df_dv = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dv)
    d2f_dv2 = (
        np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
    ) / dv**2
    phi = df_dv + 0.5 * dv * np.sign(E)[:, None] * d2f_dv2
    H_bar = np.sum(F_bar * phi, axis=1)
    rho_bar = np.fft.ifft(
        np.conj(E_multiplier) * np.fft.fft(H_bar)
    ).real
    f_bar = f_bar - dv * rho_bar[:, None]
    return f_bar, H_bar


class FullRankKSL:
    def __init__(self, f_initial, dt, H):
        # Both projectors equal I only when r = nx = nv.
        if nx != nv:
            raise ValueError("The full-rank KSL reference requires nx == nv.")
        self.f = np.array(f_initial, dtype=float, copy=True)
        self.H = np.array(H, dtype=float, copy=True)
        if self.f.shape != (nx, nv) or self.H.shape != (nx,):
            raise ValueError("State or control has an incompatible grid shape.")
        self.dt = dt
        self.rank = nx
        self.f_history = [self.f]

    def intermediate_states(self, f):
        # Exactly the original K/S substeps with both projectors equal to I.
        f_K = f + self.dt * RHS(f, self.H)
        f_S = f_K - self.dt * RHS(f_K, self.H)
        return f_K, f_S

    def step(self):
        _, f_S = self.intermediate_states(self.f)
        self.f = f_S + self.dt * RHS(f_S, self.H)
        self.f_history.append(self.f)
        return self.f

    def run(self, num_steps):
        for _ in range(num_steps):
            self.step()
        return self.f


class Adjoint:
    def __init__(self, forward):
        self.forward = forward

    def run(self):
        forward = self.forward
        h, H = forward.dt, forward.H
        # The objective's quadrature weight enters once, at the terminal value.
        f_bar = dx * dv * (forward.f - f_eq)
        J_H = np.zeros(nx)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for n in range(len(forward.f_history) - 2, -1, -1):
                f_n = forward.f_history[n]
                # Recompute two stages instead of storing three full trajectories.
                f_K, f_S = forward.intermediate_states(f_n)

                df_2, dH_2 = RHS_adjoint(f_S, H, f_bar)
                f_S_bar = f_bar + h * df_2
                df_1, dH_1 = RHS_adjoint(f_K, H, f_S_bar)
                f_K_bar = f_S_bar - h * df_1
                df_0, dH_0 = RHS_adjoint(f_n, H, f_K_bar)
                f_bar = f_K_bar + h * df_0
                # H is static and contributes at all three stages of every step.
                J_H += h * (dH_2 - dH_1 + dH_0)
        if not np.all(np.isfinite(f_bar)) or not np.all(np.isfinite(J_H)):
            raise FloatingPointError("Non-finite full-rank adjoint gradient.")
        return {"f_bar_0": f_bar, "J_H": J_H}


def gradient_a_k(J_H):
    J_H = np.asarray(J_H, dtype=float)
    if J_H.shape != (nx,):
        raise ValueError(f"J_H must have shape ({nx},).")
    return H_basis @ J_H


def objective(forward):
    return 0.5 * dx * dv * np.sum((forward.f - f_eq) ** 2)


def solve_control(a_k, dt=0.005, T=T):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(T) or T < 0:
        raise ValueError("dt must be positive and T nonnegative, both finite.")
    N = int(round(T / dt))
    forward = FullRankKSL(f_0, dt, build_H(a_k))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        forward.run(N)
        J = objective(forward)
    if not np.isfinite(J):
        raise FloatingPointError("Non-finite full-rank forward objective.")
    return forward, J


# L-BFGS uses the exact discrete adjoint of the full-matrix K/S/L update.
def optimize(a_k, dt=0.005, T=T, tol=TOL, grad_tol=GRAD_TOL,
             max_steps=MAX_STEPS, *, log=print):
    a_k = np.array(a_k, dtype=float, copy=True)
    rank = nx
    forward, J = solve_control(a_k, dt, T)
    J_history = [J]
    grad_history = []
    step_norm_history = []
    previous_a = a_k.copy()
    latest_gradient = None
    initial_a = a_k.copy()
    pending_forward = forward
    steps = 0
    stop_reason = "Max steps reached"
    log(f"[rank={rank}] Initial: J_model={J:.10e}")

    def value_and_gradient(a):
        nonlocal latest_gradient, pending_forward
        if pending_forward is not None and np.array_equal(a, initial_a):
            trajectory = pending_forward
            value = objective(trajectory)
        else:
            trajectory, value = solve_control(a, dt, T)
        pending_forward = None
        adjoint_result = Adjoint(trajectory).run()
        latest_gradient = gradient_a_k(adjoint_result["J_H"])
        if not np.all(np.isfinite(latest_gradient)):
            raise FloatingPointError("Non-finite full-rank control gradient.")
        return value, latest_gradient

    def record_iteration(intermediate_result):
        nonlocal previous_a
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
        forward, J = solve_control(a_k, dt, T)
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
    T = (len(forward.f_history) - 1) * forward.dt
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty, finite 1D array.")
    if np.any(times < 0) or np.any(times > T + 1e-12):
        raise ValueError("Plot times must lie within [0, T].")
    indices = np.rint(times / forward.dt).astype(int)
    delta_f_plot = []
    titles = []
    for n in indices:
        f_n = forward.f_history[n]
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
        filename = Path(__file__).resolve().parent / "figures" / "Adjoint_DLR_fullrank_distribution.png"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180) #type: ignore
    plt.close(fig)
    return fig, axes


def main(dt=0.005, T=T, tol=TOL, grad_tol=GRAD_TOL, max_steps=MAX_STEPS,
         *, output_dir=None):
    import csv
    from datetime import datetime
    from pathlib import Path
    import matplotlib.pyplot as plt

    if nx != nv:
        raise ValueError("The full-rank KSL reference requires nx == nv.")
    rank = nx
    output_dir = (Path(__file__).resolve().parent if output_dir is None
                  else Path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "Adjoint_DLR_fullrank_results.csv"
    plot_file = output_dir / "figures" / "Adjoint_DLR_fullrank_convergence.png"
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"fullrank_{rank}.log"
    run_started = datetime.now().isoformat(timespec="seconds")
    a_initial = np.zeros(k)
    a_initial[0] = 1e-3
    print(f"ranks={[rank]}, k={k}, grid={nx}x{nv}, T={T:g}, dt={dt:g}, "
          f"max_steps={max_steps}", flush=True)
    print("J_model: full-rank KSL; J_ref: same full-rank objective (J_model = J_ref).",
          flush=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        def log(message):
            print(message, flush=True)
            print(message, file=log_file, flush=True)

        log(f"\n--- {run_started} | rank={rank}, k={k}, grid={nx}x{nv}, "
            f"T={T:g}, dt={dt:g}, method=L-BFGS-B, max_steps={max_steps}, "
            f"tol={tol:g}, gtol={grad_tol:g}, ftol={LBFGS_FTOL:g}, "
            f"maxcor={LBFGS_MAXCOR}, maxls={LBFGS_MAXLS}, "
            "scheme=fullrank-matrix-KSL ---")
        result = None
        J_model = J_ref = np.nan
        a_final = np.full(k, np.nan)
        steps = 0
        try:
            result = optimize(a_initial, dt=dt, T=T, tol=tol,
                              grad_tol=grad_tol, max_steps=max_steps, log=log)
            J_model = J_ref = objective(result["forward"])
            result["J_history"][-1] = J_model
            a_final = result["a_k"]
            steps = result["steps"]
            stop = result["stop_reason"]
        except FloatingPointError as error:
            stop = f"Full-rank solve failed: {error}"
        row = dict(rank=rank, k=k, T=T, dt=dt, steps=steps,
                   J_model=J_model, J_ref=J_ref, stop=stop)
        row.update({f"a_{i + 1}": value for i, value in enumerate(a_final)})
        log(f"rank={rank:3d} | steps={steps:2d} | J_model={J_model:.10e} | "
            f"J_ref={J_ref:.10e} | {stop}")

    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    if result is not None:
        history = result["J_history"]
        ax.plot(np.arange(len(history)), history, "o-")
        if np.all(np.isfinite(history)) and np.all(history > 0):
            ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("J_model = J_ref (full-rank KSL)")
    ax.set_title(f"L-BFGS-B | rank={rank}, k={k}, T={T:g}, dt={dt:g}")
    ax.grid(True, alpha=0.3)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)
    print(f"CSV: {csv_file}\nPlot: {plot_file}\nLogs: {log_path}")
    return [row]


if __name__ == "__main__":
    result = main()
