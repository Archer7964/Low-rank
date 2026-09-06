# Low-Rank Methods

This two-person project studies dynamical low-rank (DLR) methods for kinetic equations, with emphasis on projector-splitting integrators for the Vlasov-Poisson equation and on external-field control of plasma instabilities.

## Project layout

### [Archer](Archer/)

Archer's work contains fixed-rank K-S-L reproductions for 1D1V and 3D3V Vlasov-Poisson problems, together with full-grid semi-Lagrangian and discrete-adjoint experiments for optimizing an external electric field. The folder also includes the source papers and the corresponding numerical figures.

### [Dylan](Dylan/)

Dylan's work implements the K-S-L projector-splitting method for a model relaxation equation and for the 1D1V Vlasov-Poisson equation. It includes central- and upwind-difference variants, reusable finite-difference and RK4 utilities, rank selection, and figures from the numerical tests. See [`Dylan/TABLE_OF_CONTENTS.txt`](Dylan/TABLE_OF_CONTENTS.txt) for the file-by-file index.

### [Some notes](<Some notes/>)

This folder collects source papers and handwritten study notes supporting the project. The notes cover the original low-rank method, the choice between discretize-first and DLR-first formulations, stability of robust DLR schemes, and optimization-based suppression of Vlasov-Poisson instabilities.
