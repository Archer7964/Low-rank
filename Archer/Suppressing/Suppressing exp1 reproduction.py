import numpy as np

nx = nv = 128
a = 0.2
b = 2 * np.pi

# [0, 4π] × [-6, 6]
dx = 4 * np.pi / nx
dv = 12 / nv
x = (np.arange(nx) + 0.5) * dx
v = -6 + (np.arange(nv) + 0.5) * dv

chi = np.exp(-a * (x - b)**2) * np.sin(x / 2)**2
f_0 = chi[:, None] * np.exp(-v[None, :]**2 / 2) / (2 * np.pi)

K = np.arange(1, 11)
H_basis = np.sin(K[:, None] * x[None, :] / 2)
TOL = 5e-5
MAX_STEPS = 100


def build_H(a_k):
    return a_k @ H_basis


def objective(f):
    return 0.5 * dx * dv * np.sum((f[-1] - f_0)**2)


def solve_E(f):
    rho_f = np.sum(f, axis=1) * dv
    q = np.ones(nx)
    q -= rho_f
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    multiplier = np.zeros(nx, dtype=complex)
    multiplier[k != 0] = -1j / k[k != 0]
    return np.fft.ifft(multiplier * np.fft.fft(q)).real


class VPSolver:
    def __init__(self, dt=0.5, T=20):
        self.dt = dt
        self.T = T
        self.nt = int(round(T / dt))

        # -v*dt/(2*dx) = n(v) + alpha(v)
        shift_x = -v * dt / (2 * dx)
        self.n_v = np.floor(shift_x).astype(int)
        self.alpha_v = shift_x - self.n_v
        self.A_v = self.build_A_v()

    def build_A_v(self):
        A_v = np.zeros((nv, nx, nx))
        k = np.arange(nx)

        for j in range(nv):
            A_v[j, k, (k + self.n_v[j]) % nx] = 1 - self.alpha_v[j]
            A_v[j, k, (k + self.n_v[j] + 1) % nx] = self.alpha_v[j]

        return A_v

    def half_x(self, f):
        f_half = np.empty_like(f)

        for j in range(nv):
            f_half[:, j] = self.A_v[j] @ f[:, j]

        return f_half

    def build_B_E(self, E, H):
        shift_v = (E + H) * self.dt / dv
        n_E = np.floor(shift_v).astype(int)
        alpha_E = shift_v - n_E

        B_E = np.zeros((nx, nv, nv))
        j = np.arange(nv)

        for i in range(nx):
            l_0 = j + n_E[i]
            l_1 = l_0 + 1
            valid_0 = (0 <= l_0) & (l_0 < nv)
            valid_1 = (0 <= l_1) & (l_1 < nv)
            B_E[i, j[valid_0], l_0[valid_0]] = 1 - alpha_E[i]
            B_E[i, j[valid_1], l_1[valid_1]] = alpha_E[i]
        return B_E

    def whole_v(self, f, E, H):
        B_E = self.build_B_E(E, H)
        f_whole = np.empty_like(f)

        for i in range(nx):
            f_whole[i, :] = B_E[i] @ f[i, :]

        return f_whole

    def step(self, f, H):
        f_star = self.half_x(f)
        E_star = solve_E(f_star)
        f_starstar = self.whole_v(f_star, E_star, H)
        f_next = self.half_x(f_starstar)
        return f_next, f_star, f_starstar, E_star

    def solve(self, f_0, H):
        f = np.empty((self.nt + 1, nx, nv))
        self.f_star = np.empty((self.nt, nx, nv))
        self.f_starstar = np.empty((self.nt, nx, nv))
        self.E_star = np.empty((self.nt, nx))
        f[0] = f_0

        for n in range(self.nt):
            (
                f[n + 1],
                self.f_star[n],
                self.f_starstar[n],
                self.E_star[n],
            ) = self.step(f[n], H)

        return f


class AdjointSolver:
    def __init__(self, forward_solver):
        self.forward_solver = forward_solver
        self.nt = forward_solver.nt

    def half_x(self, g):
        g_half = np.empty_like(g)

        for j in range(nv):
            g_half[:, j] = self.forward_solver.A_v[j].T @ g[:, j]

        return g_half

    def whole_v(self, g, f_star, E_star, H):
        dt = self.forward_solver.dt
        a_n = E_star + H
        shift_v = a_n * dt / dv
        m_n = np.floor(shift_v).astype(int)
        beta_n = shift_v - m_n

        g_star = np.empty_like(g)
        phi = np.empty_like(f_star)
        j = np.arange(nv)

        for i in range(nx):
            j_0 = j - m_n[i]
            j_1 = j_0 - 1
            valid_0 = (0 <= j_0) & (j_0 < nv)
            valid_1 = (0 <= j_1) & (j_1 < nv)

            g_0 = np.zeros(nv)
            g_1 = np.zeros(nv)
            g_0[valid_0] = g[i, j_0[valid_0]]
            g_1[valid_1] = g[i, j_1[valid_1]]

            g_star[i] = (1 - beta_n[i]) * g_0 + beta_n[i] * g_1
            phi[i] = f_star[i] * (g_0 - g_1)

        E_phi = solve_E(phi)
        g_star += dt / dv * E_phi[:, None]
        return g_star, phi, a_n, m_n, beta_n

    def step(self, g_next, n, H):
        g_starstar = self.half_x(g_next)

        f_star = self.forward_solver.f_star[n]
        E_star = self.forward_solver.E_star[n]
        g_star, phi, a_n, m_n, beta_n = self.whole_v(
            g_starstar, f_star, E_star, H
        )

        g_n = self.half_x(g_star)
        return g_n, g_star, g_starstar, phi, a_n, m_n, beta_n

    def solve(self, f, H, g_N):
        g = np.empty_like(f)
        self.g_star = np.empty((self.nt, nx, nv))
        self.g_starstar = np.empty((self.nt, nx, nv))
        self.phi = np.empty((self.nt, nx, nv))
        self.a_n = np.empty((self.nt, nx))
        self.m_n = np.empty((self.nt, nx), dtype=int)
        self.beta_n = np.empty((self.nt, nx))
        g[-1] = g_N

        for n in range(self.nt - 1, -1, -1):
            (
                g[n],
                self.g_star[n],
                self.g_starstar[n],
                self.phi[n],
                self.a_n[n],
                self.m_n[n],
                self.beta_n[n],
            ) = self.step(g[n + 1], n, H)

        return g

    def gradient_H(self):
        dt = self.forward_solver.dt
        J_H = np.zeros(nx)
        j = np.arange(nv)

        for n in range(self.nt):
            for i in range(nx):
                j_0 = j + self.m_n[n, i]
                j_1 = j_0 + 1
                valid_0 = (0 <= j_0) & (j_0 < nv)
                valid_1 = (0 <= j_1) & (j_1 < nv)

                f_0_shift = np.zeros(nv)
                f_1_shift = np.zeros(nv)
                f_0_shift[valid_0] = self.forward_solver.f_star[
                    n, i, j_0[valid_0]
                ]
                f_1_shift[valid_1] = self.forward_solver.f_star[
                    n, i, j_1[valid_1]
                ]

                J_H[i] += dx * dt * np.sum(
                    (f_1_shift - f_0_shift) * self.g_starstar[n, i]
                )

        return J_H


def optimization_step(a_k, h):
    H = build_H(a_k)
    forward_solver = VPSolver()
    f = forward_solver.solve(f_0, H)

    g_N = f[-1] - f_0
    adjoint_solver = AdjointSolver(forward_solver)
    g = adjoint_solver.solve(f, H, g_N)

    J = objective(f)
    J_H = adjoint_solver.gradient_H()
    J_a = H_basis @ J_H
    a_k = a_k - h * J_a
    return a_k, J, J_H, J_a, f, g


def optimize(a_k, h=100.0, tol=TOL, max_steps=MAX_STEPS):
    J_history = []

    for step in range(1, max_steps + 1):
        a_next, J, _, J_a, f, _ = optimization_step(a_k, h)
        J_history.append(J)

        if np.linalg.norm(J_a) <= tol:
            return a_k, np.array(J_history), f, step

        a_k = a_next

    forward_solver = VPSolver()
    f = forward_solver.solve(f_0, build_H(a_k))
    J_history.append(objective(f))
    return a_k, np.array(J_history), f, max_steps


def plot_f(f, dt=0.5):
    import matplotlib.pyplot as plt

    times = np.array([0, 10, 20])
    indices = np.rint(times / dt).astype(int)
    f_plot = f[indices]
    vmin = np.min(f_plot)
    vmax = np.max(f_plot)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)

    for ax, f_n, t in zip(axes, f_plot, times):
        image = ax.pcolormesh(
            x, v, f_n.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"T = {t}")
        ax.set_xlabel("x")

    axes[0].set_ylabel("v")
    fig.colorbar(image, ax=axes, label="f(x, v)")
    plt.show()
    return fig, axes


def main(h=100.0, tol=TOL, max_steps=MAX_STEPS):
    a_k = np.array([3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a_k, J_history, f, steps = optimize(a_k, h, tol, max_steps)
    plot_f(f)
    return a_k, J_history, f, steps


if __name__ == "__main__":
    a_k, J_history, f, steps = main()
    print(f"steps = {steps}, J = {J_history[-1]:.8e}, a_k = {a_k}")
