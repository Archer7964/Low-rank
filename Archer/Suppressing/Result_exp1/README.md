# Experiment 1 Results: Beam Shaping

These figures come from the setup in `../Suppressing exp1 reproduction.py`. Each image shows the phase-space density at T = 0, 10, and 20, while the filename records which sine modes were available to the optimized external field.

| File | Result shown |
| --- | --- |
| `K={1, 2}.png` | With only two control modes, the two initial density peaks develop strong crossing filaments and a central spiral. The control space is too limited to preserve the original shape well. |
| `K={1~5}.png` | Five modes retain the two dominant peaks much more clearly, although faint diagonal tails and a velocity shift remain. |
| `K={1~10}.png` | Ten modes give the best visual shape preservation in this comparison: the two peaks remain near their initial locations, with only weaker residual filaments. |

The current script is configured for the 10-mode case. Reproducing the other two images requires changing `K`, rebuilding `H_basis`, and using a coefficient vector of the matching length. The images provide a qualitative comparison; objective histories and control coefficients are not stored here.
