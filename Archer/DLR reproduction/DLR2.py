import numpy as np
from PIL import Image, ImageDraw


nx = 64
nv = 256
x = np.linspace(0.0, 4.0 * np.pi, nx, endpoint=False)
v = np.linspace(-6.0, 6.0, nv, endpoint=False)
dx = x[1] - x[0]
dv = v[1] - v[0]
W_x = dx * np.eye(nx)
W_v = dv * np.eye(nv)

# f_0 is a sum of two different space-velocity products
f0 = (
    (1.0 - 0.3) * np.exp(-0.5 * v[None, :] ** 2)
    * (1.0 + 0.01 * np.cos(0.5 * x[:, None]))
    + 0.3 * np.exp(-0.5 * (v[None, :] - 4.0) ** 2)
    * (1.0 + 0.01 * np.sin(0.5 * x[:, None]))
) / np.sqrt(2.0 * np.pi)

# f0 = U_0 @ S_0 @ V_0.T
rank = 5
U_full, singular_values, V_full_T = np.linalg.svd(f0, full_matrices=False)
U_0 = U_full[:, :rank] / np.sqrt(dx)
S_0 = np.sqrt(dx * dv) * np.diag(singular_values[:rank])
V_0 = V_full_T[:rank, :].T / np.sqrt(dv)


def solve_E(U, S, V):
    dx = x[1] - x[0]
    dv = v[1] - v[0]

    f = U @ S @ V.T
    rho = 1.0 - dv * np.sum(f, axis=1)
    rho = rho - np.mean(rho)

    poisson_matrix = np.zeros((nx, nx))
    for i in range(nx):
        poisson_matrix[i, i] = 2.0 / dx**2
        poisson_matrix[i, (i - 1) % nx] = -1.0 / dx**2
        poisson_matrix[i, (i + 1) % nx] = -1.0 / dx**2

    poisson_matrix[-1, :] = 1.0
    rho[-1] = 0.0
    phi = np.linalg.solve(poisson_matrix, rho)

    E = -(np.roll(phi, -1) - phi) / dx
    return E



# K-S-L steps for DLR
class DLR:
    def __init__(self, U, S, V, dt):
        self.U = U
        self.S = S
        self.V = V
        self.dt = dt

    def K_step(self):
        dx = x[1] - x[0]
        dv = v[1] - v[0]

        E = solve_E(self.U, self.S, self.V)
        K = self.U @ self.S

        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)

        dV_dv_positive = (
            np.roll(self.V, -1, axis=0) - self.V
        ) / dv
        dV_dv_negative = (
            self.V - np.roll(self.V, 1, axis=0)
        ) / dv


        #Upwind discretization
        v_positive = np.maximum(v, 0.0)
        v_negative = np.minimum(v, 0.0)

        self.c1_positive = (
            dv * self.V.T @ (v_positive[:, None] * self.V)
        )
        self.c1_negative = (
            dv * self.V.T @ (v_negative[:, None] * self.V)
        )
        self.c1 = self.c1_positive + self.c1_negative
        self.c2_positive = dv * self.V.T @ dV_dv_positive
        self.c2_negative = dv * self.V.T @ dV_dv_negative
        self.c2 = self.c2_positive

        dK_dx_positive = (K - np.roll(K, 1, axis=0)) / dx
        dK_dx_negative = (np.roll(K, -1, axis=0) - K) / dx

        K_t = (
            -dK_dx_positive @ self.c1_positive.T
            -dK_dx_negative @ self.c1_negative.T
            + E_positive[:, None] * (K @ self.c2_positive.T)
            + E_negative[:, None] * (K @ self.c2_negative.T)
        )
        K_next = K + self.dt * K_t

        U_standard, S_standard = np.linalg.qr(K_next, mode="reduced")
        self.U = U_standard / np.sqrt(dx)
        self.S = np.sqrt(dx) * S_standard
        return self.U, self.S, self.V

    def S_step(self):
        dx = x[1] - x[0]
        E = solve_E(self.U, self.S, self.V)
        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)

        dU_dx_positive = (
            self.U - np.roll(self.U, 1, axis=0)
        ) / dx
        dU_dx_negative = (
            np.roll(self.U, -1, axis=0) - self.U
        ) / dx

        self.d1_positive = (
            dx * self.U.T @ (E_positive[:, None] * self.U)
        )
        self.d1_negative = (
            dx * self.U.T @ (E_negative[:, None] * self.U)
        )
        self.d1 = self.d1_positive + self.d1_negative
        self.d2_positive = dx * self.U.T @ dU_dx_positive
        self.d2_negative = dx * self.U.T @ dU_dx_negative
        self.d2 = self.d2_negative

        S_t = (
            self.d2_positive @ self.S @ self.c1_positive.T
            + self.d2_negative @ self.S @ self.c1_negative.T
            - self.d1_positive @ self.S @ self.c2_positive.T
            - self.d1_negative @ self.S @ self.c2_negative.T
        )
        self.S = self.S + self.dt * S_t
        return self.U, self.S, self.V

    def L_step(self):
        dx = x[1] - x[0]
        dv = v[1] - v[0]

        E = solve_E(self.U, self.S, self.V)
        E_positive = np.maximum(E, 0.0)
        E_negative = np.minimum(E, 0.0)

        self.d1_positive = (
            dx * self.U.T @ (E_positive[:, None] * self.U)
        )
        self.d1_negative = (
            dx * self.U.T @ (E_negative[:, None] * self.U)
        )
        self.d1 = self.d1_positive + self.d1_negative

        L = self.V @ self.S.T
        dL_dv_E_positive = (np.roll(L, -1, axis=0) - L) / dv
        dL_dv_E_negative = (L - np.roll(L, 1, axis=0)) / dv

        v_positive = np.maximum(v, 0.0)
        v_negative = np.minimum(v, 0.0)

        L_t = (
            dL_dv_E_positive @ self.d1_positive.T
            + dL_dv_E_negative @ self.d1_negative.T
            - v_positive[:, None] * (L @ self.d2_positive.T)
            - v_negative[:, None] * (L @ self.d2_negative.T)
        )
        L_next = L + self.dt * L_t

        V_standard, S_T_standard = np.linalg.qr(L_next, mode="reduced")
        self.V = V_standard / np.sqrt(dv)
        self.S = (np.sqrt(dv) * S_T_standard).T
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



# Canvas
def save_phase_space_panels(
    data, filename, value_min, value_max, title, palette
):
    panel_width = 400
    panel_height = 300
    left_margin = 32
    panel_top = 55
    canvas = Image.new(
        "RGB",
        (left_margin + len(data) * panel_width, panel_height + 80),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((left_margin, 5), title, fill="black")
    draw.text(
        (left_margin, 22),
        f"common color scale: [{value_min:.3e}, {value_max:.3e}]",
        fill="black",
    )
    value_range = value_max - value_min
    for panel_number, (time, f) in enumerate(data):
        values = f.T
        if value_range == 0.0:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = (values - value_min) / value_range
        normalized = np.clip(normalized, 0.0, 1.0)
        if palette == "diverging":
            red = 255.0 * np.minimum(1.0, 2.0 * normalized)
            green = 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))
            blue = 255.0 * np.minimum(1.0, 2.0 * (1.0 - normalized))
        else:
            red = 255.0 * np.ones_like(normalized)
            green = 255.0 * (1.0 - normalized)
            blue = 255.0 * (1.0 - normalized)
        colors = np.stack((red, green, blue), axis=2).astype(np.uint8)
        panel = Image.fromarray(colors).resize(
            (panel_width, panel_height),
            Image.Resampling.NEAREST,
        )
        panel = panel.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        panel_left = left_margin + panel_number * panel_width
        canvas.paste(panel, (panel_left, panel_top))
        draw.text(
            (panel_left + 10, 39),
            f"t = {time:.3f}",
            fill="black",
        )
        draw.text((panel_left, panel_top + panel_height + 5), f"{x[0]:.1f}", fill="black")
        draw.text(
            (panel_left + panel_width - 30, panel_top + panel_height + 5),
            f"{x[-1] + dx:.1f}",
            fill="black",
        )
        draw.text(
            (panel_left + panel_width // 2, panel_top + panel_height + 5),
            "x",
            fill="black",
        )
    draw.text((3, panel_top - 4), f"{v[-1] + dv:.1f}", fill="black")
    draw.text((3, panel_top + panel_height - 8), f"{v[0]:.1f}", fill="black")
    draw.text((15, panel_top + panel_height // 2), "v", fill="black")
    canvas.save(filename)

def save_E_norm_plot(times, E_norms, filename):
    width = 800
    height = 400
    left = 60
    right = 20
    top = 40
    bottom = 45

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.line((left, top, left, height - bottom), fill="black", width=1)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="black",
        width=1,
    )
    time_max = max(times)
    E_norm_max = max(E_norms)
    if time_max == 0.0:
        time_max = 1.0
    if E_norm_max == 0.0:
        E_norm_max = 1.0
    points = []
    for time, E_norm in zip(times, E_norms):
        horizontal = left + (width - left - right) * time / time_max
        vertical = height - bottom - (
            height - top - bottom
        ) * E_norm / E_norm_max
        points.append((horizontal, vertical))
    if len(points) > 1:
        draw.line(points, fill="blue", width=2)
    else:
        draw.ellipse(
            (points[0][0] - 2, points[0][1] - 2,
             points[0][0] + 2, points[0][1] + 2),
            fill="blue",
        )
    draw.text((width // 2 - 35, 10), "E L2 norm", fill="black")
    draw.text((width // 2, height - 25), "t", fill="black")
    draw.text((5, top - 5), f"{max(E_norms):.3e}", fill="black")
    draw.text((left - 5, height - bottom + 8), "0", fill="black")
    draw.text(
        (width - right - 45, height - bottom + 8),
        f"{max(times):.3f}",
        fill="black",
    )
    canvas.save(filename)





def main(dt, t_final):
    solver = DLR(U_0, S_0, V_0, dt)
    num_steps = int(t_final / dt)
    snapshot_steps = {0, num_steps // 2, num_steps}
    snapshots = [(0.0, U_0 @ S_0 @ V_0.T)]
    times = [0.0]
    E_initial = solve_E(U_0, S_0, V_0)
    E_norms = [np.sqrt(dx * np.sum(E_initial**2))]

    U_final = U_0
    S_final = S_0
    V_final = V_0

    for step_number in range(1, num_steps + 1):
        U_final, S_final, V_final = solver.step()
        time = step_number * dt
        E = solve_E(U_final, S_final, V_final)
        times.append(time)
        E_norms.append(np.sqrt(dx * np.sum(E**2)))

        if step_number in snapshot_steps:
            f = U_final @ S_final @ V_final.T
            snapshots.append((time, f))

    f_reference = np.mean(
        snapshots[0][1],
        axis=0,
        keepdims=True,
    )
    perturbations = [
        (time, f - f_reference)
        for time, f in snapshots
    ]
    color_limit = max(
        np.max(np.abs(g))
        for _, g in perturbations
    )
    save_phase_space_panels(
        perturbations,
        "DLR_three_times.png",
        -color_limit,
        color_limit,
        "Perturbation g(t,x,v) = f(t,x,v) - <f0>_x",
        "diverging",
    )

    f_min = min(np.min(f) for _, f in snapshots)
    f_max = max(np.max(f) for _, f in snapshots)
    save_phase_space_panels(
        snapshots,
        "DLR_full_distribution_three_times.png",
        f_min,
        f_max,
        "Distribution f(t,x,v)",
        "sequential",
    )
    save_E_norm_plot(times, E_norms, "DLR_E_L2.png")

    return U_final, S_final, V_final


if __name__ == "__main__":
    dt = 0.025
    t_final = 200.0
    U_final, S_final, V_final = main(dt, t_final)
