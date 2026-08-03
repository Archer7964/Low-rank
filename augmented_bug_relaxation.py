"""Simlpified Augmented BUG

Equation:
        
"""

import numpy as np
from PIL import Image, ImageDraw

NX = 64
NV = 64
FINAL_TIME = 1.0
DT = 0.01
RANK_TOL = 1.0e-10
MAX_RANK = 4


def problem_functions(x, v):
    u0 = 1.0 + 0.2 * np.cos(2.0 * np.pi * x)
    v0 = 1.0 + 0.2 * np.cos(2.0 * np.pi * v)
    a = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    b = 1.0 + 0.2 * np.sin(2.0 * np.pi * v)
    return u0, v0, a, b

def exact_solution(t, u0, v0, a, b):
    decay = np.exp(-t)
    initial_part = u0[:, None] * v0[None, :]
    source_part = a[:, None] * b[None, :]
    return decay * initial_part + (1.0 - decay) * source_part

def orth(matrix, weights):
    # return basis satisfying Q.T @ diag(weights) @ Q = I
    scaled = np.sqrt(weights)[:, None] * matrix
    Q, R = np.linalg.qr(scaled, mode="reduced")
    diagonal = np.abs(np.diag(R))
    keep = diagonal > 1.0e-12 * diagonal.max()
    return Q[:, keep] / np.sqrt(weights)[:, None]


def select_rank(singular_values):
    limit = RANK_TOL * np.linalg.norm(singular_values)
    for rank in range(1, len(singular_values) + 1):
        if np.linalg.norm(singular_values[rank:]) <= limit:
            return min(rank, MAX_RANK)
    return min(len(singular_values), MAX_RANK)


def truncate(U_hat, S_hat, V_hat):
    P, singular_values, Qt = np.linalg.svd(
        S_hat, full_matrices=False
    )
    rank = select_rank(singular_values)
    U = U_hat @ P[:, :rank]
    S = np.diag(singular_values[:rank])
    V = V_hat @ Qt.T[:, :rank]
    return U, S, V

def AugBUG(U, S, V, dt, weights_x, weights_v, a, b):
    # K-step
    K0 = U @ S
    inner_b_V = V.T @ (weights_v * b)
    K1 = K0 + dt * (-K0 + a[:, None] * inner_b_V[None, :])

    # L-step
    # L_t = -L + b(v)<a,U>_x
    L0 = V @ S.T
    inner_a_U = U.T @ (weights_x * a)
    L1 = L0 + dt * (-L0 + b[:, None] * inner_a_U[None, :])

    # Augmented basis
    U_hat = orth(np.column_stack((K1, U)), weights_x)
    V_hat = orth(np.column_stack((L1, V)), weights_v)

    M = U_hat.T @ (weights_x[:, None] * U)
    N = V_hat.T @ (weights_v[:, None] * V)
    S_hat0 = M @ S @ N.T

    # S_t = -S + <a,U_hat>_x <b,V_hat>_v^T
    inner_a_U_hat = U_hat.T @ (weights_x * a)
    inner_b_V_hat = V_hat.T @ (weights_v * b)
    source_matrix = inner_a_U_hat[:, None] * inner_b_V_hat[None, :]
    S_hat1 = S_hat0 + dt * (-S_hat0 + source_matrix)

    # SVD
    return truncate(U_hat, S_hat1, V_hat)


# MAP to show the solution and error
def color_map(scaled):
    positions = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    palette = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ])
    red = np.interp(scaled, positions, palette[:, 0])
    green = np.interp(scaled, positions, palette[:, 1])
    blue = np.interp(scaled, positions, palette[:, 2])
    return np.stack((red, green, blue), axis=2).astype(np.uint8)
def heatmap(matrix, value_min, value_max, width=300, height=240):
    scaled = (matrix - value_min) / (value_max - value_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    colors = color_map(scaled)
    colors = np.transpose(colors, (1, 0, 2))[::-1, :, :]
    return Image.fromarray(colors).resize((width, height))
def save_solution_image(numerical, exact):
    error = np.abs(numerical - exact)
    value_min = min(numerical.min(), exact.min(), error.min())
    value_max = max(numerical.max(), exact.max(), error.max())
    images = [
        heatmap(numerical, value_min, value_max),
        heatmap(exact, value_min, value_max),
        heatmap(error, value_min, value_max),
    ]
    titles = ["Low-rank solution", "Exact solution", "Absolute error"]
    panel_width = 315
    panel_height = 240
    colorbar_x = 3 * panel_width + 20
    canvas = Image.new("RGB", (colorbar_x + 90, 300), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(3):
        draw.text((index * panel_width + 10, 8), titles[index], fill="black")
        canvas.paste(images[index], (index * panel_width + 5, 30))
    gradient = np.linspace(1.0, 0.0, panel_height)[:, None]
    gradient = np.repeat(gradient, 24, axis=1)
    colorbar = Image.fromarray(color_map(gradient))
    canvas.paste(colorbar, (colorbar_x, 30))
    draw.rectangle(
        (colorbar_x, 30, colorbar_x + 24, 30 + panel_height),
        outline="black",
    )
    draw.text((colorbar_x, 8), "Shared scale", fill="black")
    for tick in range(5):
        y = 30 + tick * panel_height / 4
        value = value_max - tick * (value_max - value_min) / 4
        draw.line((colorbar_x + 24, y, colorbar_x + 30, y), fill="black")
        draw.text((colorbar_x + 34, y - 6), f"{value:.3f}", fill="black")
    draw.text(
        (2 * panel_width + 10, 278),
        f"max error = {error.max():.3e}",
        fill="black",
    )
    image_name = "augmented_bug_solution.png"
    canvas.save(image_name)
    print("Image saved as:", image_name)
    canvas.show()


def main():
    x = (np.arange(NX) + 0.5) / NX
    v = (np.arange(NV) + 0.5) / NV
    weights_x = np.full(NX, 1.0 / NX)
    weights_v = np.full(NV, 1.0 / NV)
    u0, v0, a, b = problem_functions(x, v)
    norm_u0 = np.sqrt(np.sum(weights_x * u0**2))
    norm_v0 = np.sqrt(np.sum(weights_v * v0**2))
    U = (u0 / norm_u0)[:, None]
    S = np.array([[norm_u0 * norm_v0]])
    V = (v0 / norm_v0)[:, None]

    time = 0.0
    times = [time]
    ranks = [S.shape[0]]

    # Augmented BUG
    while time < FINAL_TIME - 1.0e-14:
        dt = min(DT, FINAL_TIME - time)
        U, S, V = AugBUG(U, S, V, dt, weights_x, weights_v, a, b)
        time += dt
        times.append(time)
        ranks.append(S.shape[0])

    numerical = U @ S @ V.T
    exact = exact_solution(time, u0, v0, a, b)
    weights = weights_x[:, None] * weights_v[None, :]
    error = np.sqrt(np.sum(weights * (numerical - exact) ** 2))
    exact_norm = np.sqrt(np.sum(weights * exact**2))
    relative_error = error / exact_norm

    save_solution_image(numerical, exact)
    print("Relative L2 Error:", relative_error)


if __name__ == "__main__":
    main()
