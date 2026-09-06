# DLR Reproduction

This folder reproduces a fixed-rank projector-splitting approach for the Vlasov-Poisson equation described in *A Low-Rank Projector-Splitting Integrator for the Vlasov-Poisson Equation*.

## Code and reference

| File | Description |
| --- | --- |
| `56.pdf` | The reference paper by Lukas Einkemmer and Christian Lubich. |
| `DLR.py` | A rank-5 1D1V K-S-L solver on a periodic 64 x 256 grid. It starts from a weakly perturbed Maxwellian, uses upwind finite differences, solves Poisson's equation for the self-consistent field, and advances to T = 200. |
| `DLR2.py` | The same 1D1V solver applied to a two-component distribution: a Maxwellian near v = 0 plus a shifted component near v = 4, with different spatial perturbations. |
| `DLR_3D.py` | A rank-5 3D3V extension. It uses 64^3 position and velocity grids, periodic upwind differences in all three directions, and a Fourier Poisson solve. The saved diagnostic is a 2D slice of the six-dimensional distribution. |

Both 1D scripts write `DLR_three_times.png`, `DLR_full_distribution_three_times.png`, and `DLR_E_L2.png` to the current working directory. Running `DLR2.py` after `DLR.py` will therefore overwrite the first run's figures. The tracked versions match the single-Maxwellian test in `DLR.py`; no separately named `DLR2.py` result is currently stored.

## Results

- `DLR_three_times.png` shows the perturbation at t = 0, 100, and 200. Its spatial structure phase-mixes into increasingly fine velocity-localized bands and becomes much weaker by the final time.
- `DLR_full_distribution_three_times.png` shows that the full distribution remains close to a spatially homogeneous Maxwellian even while the smaller perturbation evolves.
- `DLR_E_L2.png` shows rapidly damped oscillations of the electric-field L2 norm, consistent with the Landau-damping test represented by `DLR.py`.
- `DLR_3D_result.png` shows the 3D3V slice at t = 0, 2.5, and 5. The tilted bands record phase-space transport, while the lower log-scale curve shows an oscillatory electric field with a decreasing envelope and progressively lower minima.

Run the scripts from this directory so that their relative output paths remain here. They require NumPy and Pillow.
