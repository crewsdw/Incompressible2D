# Incompressible2D
Custom discontinuous Galerkin / Fourier spectral solver for incompressible 2D Navier-Stokes equation

<p align="center">
<img src="https://raw.githubusercontent.com/crewsdw/Incompressible2D/master/images/k234_rs126/vorticity_viscosity1em2.png" width="400" />
<img src="https://raw.githubusercontent.com/crewsdw/Incompressible2D/master/images/k1234_rs126/viscosity1em2.png" width="400" />
</p>

## About
Vector-valued variable is fluid velocity, whose curl is computed spectrally in order to plot vorticity in post-processing.
Mass density is supposed to be a constant background.

### Objectives
Experimental objectives of this project on high-order discontinuous Galerkin methods:
1) robustly tested experimental spectral methods for the pressure Poisson equation and viscosity,
2) concisely-coded and efficient GPU implementations,

### Information
Implementation notes can be found at: https://dcrews.gitlab.io/potential-flux/papers/incompressible_euler.pdf
