import numpy as np
from PIL import Image, ImageDraw


# Each spatial and velocity direction uses the same uniform grid.
nx = 64
nv = 64
x1 = np.linspace(0.0, 4.0 * np.pi, nx, endpoint=False)
x2 = np.linspace(0.0, 4.0 * np.pi, nx, endpoint=False)
x3 = np.linspace(0.0, 4.0 * np.pi, nx, endpoint=False)
v1 = np.linspace(-6.0, 6.0, nv, endpoint=False)
v2 = np.linspace(-6.0, 6.0, nv, endpoint=False)
v3 = np.linspace(-6.0, 6.0, nv, endpoint=False)
dx = x1[1] - x1[0]
dv = v1[1] - v1[0]

X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing="ij")
V1, V2, V3 = np.meshgrid(v1, v2, v3, indexing="ij")

nx_total = nx**3
nv_total = nv**3
weight_x = dx**3
weight_v = dv**3

# Initial condition
alpha = 0.01
k = 0.5
phase = k * X1 + k * X2 + k * X3
maxwellian = np.exp(
    -0.5 * (V1**2 + V2**2 + V3**2)
) / (2.0 * np.pi) ** 1.5

space_factor = (1.0 + alpha * np.cos(phase)).reshape(nx_total)
velocity_factor = maxwellian.reshape(nv_total)
rho0 = -alpha * np.cos(phase)


rank = 5
random_generator = np.random.default_rng(0)
U_candidates = np.column_stack((
    space_factor,
    random_generator.standard_normal((nx_total, rank - 1)),
))
V_candidates = np.column_stack((
    velocity_factor,
    random_generator.standard_normal((nv_total, rank - 1)),
))

U_0, _ = np.linalg.qr(U_candidates, mode="reduced")
V_0, _ = np.linalg.qr(V_candidates, mode="reduced")
U_0 = U_0 / np.sqrt(weight_x)
V_0 = V_0 / np.sqrt(weight_v)

S_0 = np.zeros((rank, rank))
S_0[0, 0] = (
    weight_x * (U_0[:, 0] @ space_factor)
    * weight_v * (V_0[:, 0] @ velocity_factor)
)

del U_candidates, V_candidates, random_generator

wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
wave_number_squared = (
    wave_numbers[:, None, None] ** 2
    + wave_numbers[None, :, None] ** 2
    + wave_numbers[None, None, :] ** 2
)


def solve_E(U, S, V):
    velocity_integrals = weight_v * np.sum(V, axis=0)
    density = U @ (S @ velocity_integrals)
    rho = 1.0 - density
    rho = rho - np.mean(rho)
    rho = rho.reshape(nx, nx, nx)

    rho_hat = np.fft.fftn(rho)
    phi_hat = np.zeros_like(rho_hat)
    nonzero_mode = wave_number_squared > 0.0
    phi_hat[nonzero_mode] = (
        rho_hat[nonzero_mode] / wave_number_squared[nonzero_mode]
    )

    E1 = np.fft.ifftn(
        -1j * wave_numbers[:, None, None] * phi_hat
    ).real
    E2 = np.fft.ifftn(
        -1j * wave_numbers[None, :, None] * phi_hat
    ).real
    E3 = np.fft.ifftn(
        -1j * wave_numbers[None, None, :] * phi_hat
    ).real
    E = np.stack((E1, E2, E3), axis=-1)
    return E.reshape(nx_total, 3)



# K-S-L steps for DLR
class DLR:
    def __init__(self, U, S, V, dt):
        self.U = U
        self.S = S
        self.V = V
        self.dt = dt

    def K_step(self):
        E = solve_E(self.U, self.S, self.V)
        K = self.U @ self.S

        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)

        rank = self.S.shape[0]
        K_grid = K.reshape(nx, nx, nx, rank)
        V_grid = self.V.reshape(nv, nv, nv, rank)
        velocity_components = (
            V1.reshape(nv_total),
            V2.reshape(nv_total),
            V3.reshape(nv_total),
        )

        self.c_m1_positive = np.zeros((3, rank, rank))
        self.c_m1_negative = np.zeros((3, rank, rank))
        self.c_m2_positive = np.zeros((3, rank, rank))
        self.c_m2_negative = np.zeros((3, rank, rank))
        K_t = np.zeros_like(K)

        for m in range(3):
            velocity_m = velocity_components[m]
            velocity_m_positive = np.maximum(velocity_m, 0.0)
            velocity_m_negative = np.minimum(velocity_m, 0.0)

            self.c_m1_positive[m] = (
                dv**3
                * self.V.T
                @ (velocity_m_positive[:, None] * self.V)
            )
            self.c_m1_negative[m] = (
                dv**3
                * self.V.T
                @ (velocity_m_negative[:, None] * self.V)
            )

            dV_dv_positive = (
                np.roll(V_grid, -1, axis=m) - V_grid
            ) / dv
            dV_dv_negative = (
                V_grid - np.roll(V_grid, 1, axis=m)
            ) / dv
            dV_dv_positive = dV_dv_positive.reshape(nv_total, rank)
            dV_dv_negative = dV_dv_negative.reshape(nv_total, rank)

            self.c_m2_positive[m] = (
                dv**3 * self.V.T @ dV_dv_positive
            )
            self.c_m2_negative[m] = (
                dv**3 * self.V.T @ dV_dv_negative
            )

            dK_dx_positive = (
                K_grid - np.roll(K_grid, 1, axis=m)
            ) / dx
            dK_dx_negative = (
                np.roll(K_grid, -1, axis=m) - K_grid
            ) / dx
            dK_dx_positive = dK_dx_positive.reshape(nx_total, rank)
            dK_dx_negative = dK_dx_negative.reshape(nx_total, rank)

            K_t += (
                -dK_dx_positive @ self.c_m1_positive[m].T
                -dK_dx_negative @ self.c_m1_negative[m].T
                + E_positive[:, m, None]
                * (K @ self.c_m2_positive[m].T)
                + E_negative[:, m, None]
                * (K @ self.c_m2_negative[m].T)
            )

        self.c_m1 = self.c_m1_positive + self.c_m1_negative
        self.c_m2 = self.c_m2_positive + self.c_m2_negative
        K_next = K + self.dt * K_t

        U_standard, S_standard = np.linalg.qr(K_next, mode="reduced")
        self.U = U_standard / np.sqrt(dx**3)
        self.S = np.sqrt(dx**3) * S_standard
        return self.U, self.S, self.V
    
    def S_step(self):
        E = solve_E(self.U, self.S, self.V)
        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)

        rank = self.S.shape[0]
        U_grid = self.U.reshape(nx, nx, nx, rank)

        self.d_m1_positive = np.zeros((3, rank, rank))
        self.d_m1_negative = np.zeros((3, rank, rank))
        self.d_m2_positive = np.zeros((3, rank, rank))
        self.d_m2_negative = np.zeros((3, rank, rank))
        S_t = np.zeros_like(self.S)

        for m in range(3):
            self.d_m1_positive[m] = (
                dx**3
                * self.U.T
                @ (E_positive[:, m, None] * self.U)
            )
            self.d_m1_negative[m] = (
                dx**3
                * self.U.T
                @ (E_negative[:, m, None] * self.U)
            )

            dU_dx_positive = (
                U_grid - np.roll(U_grid, 1, axis=m)
            ) / dx
            dU_dx_negative = (
                np.roll(U_grid, -1, axis=m) - U_grid
            ) / dx
            dU_dx_positive = dU_dx_positive.reshape(nx_total, rank)
            dU_dx_negative = dU_dx_negative.reshape(nx_total, rank)

            self.d_m2_positive[m] = (
                dx**3 * self.U.T @ dU_dx_positive
            )
            self.d_m2_negative[m] = (
                dx**3 * self.U.T @ dU_dx_negative
            )

            S_t += (
                self.d_m2_positive[m]
                @ self.S
                @ self.c_m1_positive[m].T
                + self.d_m2_negative[m]
                @ self.S
                @ self.c_m1_negative[m].T
                - self.d_m1_positive[m]
                @ self.S
                @ self.c_m2_positive[m].T
                - self.d_m1_negative[m]
                @ self.S
                @ self.c_m2_negative[m].T
            )

        self.d_m1 = self.d_m1_positive + self.d_m1_negative
        self.d_m2 = self.d_m2_positive + self.d_m2_negative
        self.S = self.S + self.dt * S_t
        return self.U, self.S, self.V

    def L_step(self):
        E = solve_E(self.U, self.S, self.V)
        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)

        rank = self.S.shape[0]
        U_grid = self.U.reshape(nx, nx, nx, rank)
        velocity_components = (
            V1.reshape(nv_total),
            V2.reshape(nv_total),
            V3.reshape(nv_total),
        )

        self.d_m1_positive = np.zeros((3, rank, rank))
        self.d_m1_negative = np.zeros((3, rank, rank))
        self.d_m2_positive = np.zeros((3, rank, rank))
        self.d_m2_negative = np.zeros((3, rank, rank))

        for m in range(3):
            self.d_m1_positive[m] = (
                dx**3
                * self.U.T
                @ (E_positive[:, m, None] * self.U)
            )
            self.d_m1_negative[m] = (
                dx**3
                * self.U.T
                @ (E_negative[:, m, None] * self.U)
            )

            dU_dx_positive = (
                U_grid - np.roll(U_grid, 1, axis=m)
            ) / dx
            dU_dx_negative = (
                np.roll(U_grid, -1, axis=m) - U_grid
            ) / dx
            dU_dx_positive = dU_dx_positive.reshape(nx_total, rank)
            dU_dx_negative = dU_dx_negative.reshape(nx_total, rank)

            self.d_m2_positive[m] = (
                dx**3 * self.U.T @ dU_dx_positive
            )
            self.d_m2_negative[m] = (
                dx**3 * self.U.T @ dU_dx_negative
            )

        self.d_m1 = self.d_m1_positive + self.d_m1_negative
        self.d_m2 = self.d_m2_positive + self.d_m2_negative

        L = self.V @ self.S.T
        L_grid = L.reshape(nv, nv, nv, rank)
        L_t = np.zeros_like(L)

        for m in range(3):
            dL_dv_positive = (
                np.roll(L_grid, -1, axis=m) - L_grid
            ) / dv
            dL_dv_negative = (
                L_grid - np.roll(L_grid, 1, axis=m)
            ) / dv
            dL_dv_positive = dL_dv_positive.reshape(nv_total, rank)
            dL_dv_negative = dL_dv_negative.reshape(nv_total, rank)

            velocity_m = velocity_components[m]
            velocity_m_positive = np.maximum(velocity_m, 0.0)
            velocity_m_negative = np.minimum(velocity_m, 0.0)

            L_t += (
                dL_dv_positive @ self.d_m1_positive[m].T
                + dL_dv_negative @ self.d_m1_negative[m].T
                - velocity_m_positive[:, None]
                * (L @ self.d_m2_positive[m].T)
                - velocity_m_negative[:, None]
                * (L @ self.d_m2_negative[m].T)
            )

        L_next = L + self.dt * L_t

        V_standard, S_T_standard = np.linalg.qr(L_next, mode="reduced")
        self.V = V_standard / np.sqrt(dv**3)
        self.S = (np.sqrt(dv**3) * S_T_standard).T

        V_grid = self.V.reshape(nv, nv, nv, rank)
        self.c_m1_positive = np.zeros((3, rank, rank))
        self.c_m1_negative = np.zeros((3, rank, rank))
        self.c_m2_positive = np.zeros((3, rank, rank))
        self.c_m2_negative = np.zeros((3, rank, rank))

        for m in range(3):
            velocity_m = velocity_components[m]
            velocity_m_positive = np.maximum(velocity_m, 0.0)
            velocity_m_negative = np.minimum(velocity_m, 0.0)

            self.c_m1_positive[m] = (
                dv**3
                * self.V.T
                @ (velocity_m_positive[:, None] * self.V)
            )
            self.c_m1_negative[m] = (
                dv**3
                * self.V.T
                @ (velocity_m_negative[:, None] * self.V)
            )

            dV_dv_positive = (
                np.roll(V_grid, -1, axis=m) - V_grid
            ) / dv
            dV_dv_negative = (
                V_grid - np.roll(V_grid, 1, axis=m)
            ) / dv
            dV_dv_positive = dV_dv_positive.reshape(nv_total, rank)
            dV_dv_negative = dV_dv_negative.reshape(nv_total, rank)

            self.c_m2_positive[m] = (
                dv**3 * self.V.T @ dV_dv_positive
            )
            self.c_m2_negative[m] = (
                dv**3 * self.V.T @ dV_dv_negative
            )

        self.c_m1 = self.c_m1_positive + self.c_m1_negative
        self.c_m2 = self.c_m2_positive + self.c_m2_negative

        E = solve_E(self.U, self.S, self.V)
        self.E = E
        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)
        for m in range(3):
            self.d_m1_positive[m] = (
                dx**3
                * self.U.T
                @ (E_positive[:, m, None] * self.U)
            )
            self.d_m1_negative[m] = (
                dx**3
                * self.U.T
                @ (E_negative[:, m, None] * self.U)
            )
        self.d_m1 = self.d_m1_positive + self.d_m1_negative

        return self.U, self.S, self.V

    def step(self):
        self.K_step()
        self.S_step()
        self.L_step()
        return self.U, self.S, self.V

    def run(self, num_steps):
        for _ in range(num_steps):
            self.step()

        return self.U, self.S, self.V


def phase_space_slice(U, S, V):
    """Return f(x1, x2=0, x3=0, v1, v2=0, v3=0)."""
    x_indices = np.arange(nx) * nx**2
    zero_velocity = np.argmin(np.abs(v1))
    v_indices = (
        np.arange(nv) * nv**2
        + zero_velocity * nv
        + zero_velocity
    )
    return U[x_indices] @ S @ V[v_indices].T


def save_result_plot(snapshots, times, E_norms, filename):
    panel_width = 360
    panel_height = 240
    panel_gap = 20
    panel_left = 60
    panel_top = 70
    width = 1240
    height = 720

    reference = np.mean(snapshots[0][1], axis=0, keepdims=True)
    perturbations = [
        (time, f_slice - reference)
        for time, f_slice in snapshots
    ]
    color_limit = max(
        np.max(np.abs(perturbation))
        for _, perturbation in perturbations
    )

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (panel_left, 10),
        "Perturbation slice: f(t,x1,0,0,v1,0,0) - reference",
        fill="black",
    )
    draw.text(
        (panel_left, 28),
        f"common color scale: [-{color_limit:.3e}, {color_limit:.3e}]",
        fill="black",
    )

    for panel_number, (time, perturbation) in enumerate(perturbations):
        values = perturbation.T
        if color_limit == 0.0:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = 0.5 + 0.5 * values / color_limit
        normalized = np.clip(normalized, 0.0, 1.0)

        red = 255.0 * np.minimum(1.0, 2.0 * normalized)
        green = 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))
        blue = 255.0 * np.minimum(1.0, 2.0 * (1.0 - normalized))
        colors = np.stack((red, green, blue), axis=2).astype(np.uint8)
        panel = Image.fromarray(colors).resize(
            (panel_width, panel_height),
            Image.Resampling.NEAREST,
        )
        panel = panel.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        left = panel_left + panel_number * (panel_width + panel_gap)
        canvas.paste(panel, (left, panel_top))
        draw.rectangle(
            (left, panel_top, left + panel_width, panel_top + panel_height),
            outline="black",
        )
        draw.text((left + 8, panel_top - 18), f"t = {time:.3f}", fill="black")
        draw.text((left, panel_top + panel_height + 6), f"{x1[0]:.1f}", fill="black")
        draw.text(
            (left + panel_width - 28, panel_top + panel_height + 6),
            f"{x1[-1] + dx:.1f}",
            fill="black",
        )
        draw.text(
            (left + panel_width // 2, panel_top + panel_height + 6),
            "x1",
            fill="black",
        )

    draw.text((8, panel_top - 4), f"{v1[-1] + dv:.1f}", fill="black")
    draw.text((8, panel_top + panel_height - 8), f"{v1[0]:.1f}", fill="black")
    draw.text((25, panel_top + panel_height // 2), "v1", fill="black")

    chart_left = 85
    chart_right = width - 50
    chart_top = 415
    chart_bottom = height - 55
    draw.text(
        (chart_left, chart_top - 28),
        "Electric-field L2 norm (log scale)",
        fill="black",
    )

    log_norms = np.log10(
        np.maximum(np.asarray(E_norms), np.finfo(float).tiny)
    )
    y_min = float(np.min(log_norms))
    y_max = float(np.max(log_norms))
    if y_max == y_min:
        y_min -= 0.5
        y_max += 0.5
    else:
        padding = 0.05 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    time_max = max(times)
    if time_max == 0.0:
        time_max = 1.0

    for tick_number in range(5):
        fraction = tick_number / 4.0
        vertical = chart_bottom - fraction * (chart_bottom - chart_top)
        level = y_min + fraction * (y_max - y_min)
        draw.line(
            (chart_left, vertical, chart_right, vertical),
            fill=(225, 225, 225),
        )
        draw.text((8, vertical - 6), f"{10.0**level:.1e}", fill="black")

    draw.line(
        (chart_left, chart_top, chart_left, chart_bottom),
        fill="black",
    )
    draw.line(
        (chart_left, chart_bottom, chart_right, chart_bottom),
        fill="black",
    )
    points = []
    for time, log_norm in zip(times, log_norms):
        horizontal = chart_left + (
            (chart_right - chart_left) * time / time_max
        )
        vertical = chart_bottom - (
            (chart_bottom - chart_top)
            * (log_norm - y_min)
            / (y_max - y_min)
        )
        points.append((horizontal, vertical))
    if len(points) > 1:
        draw.line(points, fill=(20, 80, 200), width=2)
    elif points:
        horizontal, vertical = points[0]
        draw.ellipse(
            (horizontal - 2, vertical - 2, horizontal + 2, vertical + 2),
            fill=(20, 80, 200),
        )

    draw.text((chart_left - 4, chart_bottom + 8), "0", fill="black")
    draw.text(
        (chart_right - 38, chart_bottom + 8),
        f"{max(times):.3f}",
        fill="black",
    )
    draw.text(
        ((chart_left + chart_right) // 2, chart_bottom + 8),
        "t",
        fill="black",
    )
    canvas.save(filename)



def main(dt, t_final):
    solver = DLR(U_0, S_0, V_0, dt)
    num_steps = int(t_final / dt)
    snapshot_steps = {0, num_steps // 2, num_steps}
    snapshots = [(0.0, phase_space_slice(U_0, S_0, V_0))]
    times = [0.0]
    E_initial = solve_E(U_0, S_0, V_0)
    E_norms = [np.sqrt(weight_x * np.sum(E_initial**2))]

    U_final = U_0
    S_final = S_0
    V_final = V_0
    for step_number in range(1, num_steps + 1):
        U_final, S_final, V_final = solver.step()
        time = step_number * dt
        times.append(time)
        E_norms.append(np.sqrt(weight_x * np.sum(solver.E**2)))

        if step_number in snapshot_steps:
            snapshots.append((
                time,
                phase_space_slice(U_final, S_final, V_final),
            ))

    save_result_plot(
        snapshots,
        times,
        E_norms,
        "DLR_3D_result.png",
    )
    return U_final, S_final, V_final


if __name__ == "__main__":
    dt = 0.025
    t_final = 5.0
    U_final, S_final, V_final = main(dt, t_final)
