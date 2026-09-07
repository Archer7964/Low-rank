# Projector Splitter Integrator (PSI) for the Vlasov-Poisson (VP) Equation

## Code
| File | Description |
| --- | --- |
| **Central Difference Scheme** | |
| `equations.py` | KSL steps and electrical field solver using Simpson's integration
| `ksl_vp_eq.py` | VP equation solver using PSI |
| **Upwind Difference Scheme** | |
| `upwind_equations.py` | KSL steps using the upwind scheme and electrical field solver |
| `ksl_upwind_vpeq.py` | VP equation solver using PSI and the upwind scheme *(not working)*|

## Results
| File | Description |
| --- | --- |
| **Central Difference Scheme** | |
| `ksl_vpeq_figure1.png` | Evolution of the non-stable two-stream VP problem in 1D |
| `ksl_vpeq_figure2.png` | Evolution of a non-stable two-point VP problem in 1D |
| **Upwind Difference Scheme** | |
| `ksl_upwind_vpeq_figure1.png` | Evolution of a non-stable two-point VP problem in 1D (under upwind difference) |
