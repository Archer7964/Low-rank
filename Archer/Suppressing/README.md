# Suppressing Vlasov-Poisson Instability

This folder reproduces experiments from *Suppressing Instability in a Vlasov-Poisson System by an External Electric Field through Constrained Optimization*. The numerical workflow uses Strang-type semi-Lagrangian transport for the forward problem, a matching discrete adjoint for gradients, and a Fourier representation of the static control field.

## Code and reference

| File | Description |
| --- | --- |
| `Suppressing.pdf` | The reference paper by Lukas Einkemmer, Qin Li, Li Wang, and Yunan Yang. |
| `Suppressing exp1 reproduction.py` | Reproduces the beam-shaping experiment on a 128 x 128 grid up to T = 20. It minimizes the final-time distance from the initial profile using sine modes sin(kx/2) and a discrete-adjoint gradient descent. The current file is configured for k = 1, ..., 10. |
| `2-stream Instability.py` | Reproduces a symmetric two-stream control problem with beams centered at v = +/-2.4. Five cosine modes parameterize the external field, and an adjoint gradient with backtracking line search minimizes the distance from the homogeneous two-stream equilibrium at T = 40. |
| `2-stream for DLR2.py` | Applies the same full-grid control solver to the asymmetric two-component initial condition used in `DLR2.py`, using five cosine and five sine modes. Despite its name, this script does not use a low-rank factorization; it is a full phase-space comparison case. No separately identified result images are stored for it. |

## Result folders

- [`Result_exp1`](Result_exp1/) compares beam-shaping results obtained with 2, 5, and 10 control modes.
- [`Result_exp2`](Result_exp2/) contains the distribution and electric-energy diagnostics for the symmetric two-stream experiment.

The plotting functions currently call `plt.show()` but do not call `savefig`. Reproducing the tracked PNG files therefore requires exporting the displayed figures or adding an explicit save command. The scripts require NumPy and Matplotlib.
