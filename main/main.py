import numpy as np
import grid as g
import basis as b
import elliptic as ell
import fluxes as fx
import timestep as ts
import plotter as my_plt

# Parameters
order = 8
res_x, res_y = 25, 25
nu = 1.0e-2

# Flags
plot_IC = True
experimental_viscosity = True

# Build basis
print('Initializing basis...')
orders = np.array([order, order])
basis = b.Basis2D(orders)

# Initialize grids
print('\nInitializing grids...')
L = 2.0 * np.pi
print('Domain length is ' + str(L))
lows = np.array([-L / 2.0, -L / 2.0])
highs = np.array([L / 2.0, L / 2.0])
resolutions = np.array([res_x, res_y])
resolutions_ghosts = np.array([res_x + 2, res_y + 2])
grids = g.Grid2D(basis=basis, lows=lows, highs=highs, resolutions=resolutions, linspace=True)

# Time info
final_time = 2.0
write_time = 0.05

# Initialize vector-valued variable
velocity = g.Vector(resolutions=resolutions_ghosts, orders=orders)
velocity.initialize(grids=grids, numbers=[1, 2, 3, 4])

# Test elliptic class and pressure solve
elliptic = ell.Elliptic(grids=grids)
elliptic.pressure_solve(velocity=velocity, grids=grids)

# Initialize plotter class for visualization
plotter = my_plt.Plotter2D(grids=grids)

print('\nVisualizing initial condition...')
if plot_IC:
    plotter.vector_contourf(vector=velocity)
    plotter.scalar_contourf(scalar=elliptic.pressure)
    plotter.streamlines2d(vector=velocity, grids=grids)
    plotter.vorticity_contourf(vector_arr=velocity.arr, grids=grids)
    plotter.show()

# Try solution to some time
# dg_flux = fx.DGFlux(resolutions=resolutions_ghosts, orders=orders,
#                     experimental_viscosity=experimental_viscosity, nu=nu)
dg_flux = fx.Spectral(resolutions=resolutions_ghosts, orders=orders, nu=nu)
stepper = ts.Stepper(time_order=3, space_order=order,
                     write_time=write_time, final_time=final_time)

print('\nBeginning main loop...')
stepper.main_loop(vector=velocity, basis=basis, elliptic=elliptic, grids=grids, dg_flux=dg_flux)

print('\nVisualizing stop time condition...')
if plot_IC:
    plotter.vector_contourf(vector=velocity)
    plotter.scalar_contourf(scalar=elliptic.pressure)
    plotter.streamlines2d(vector=velocity, grids=grids)
    plotter.vorticity_contourf(vector_arr=velocity.arr, grids=grids)

    # make movie
    plotter.animate2d(stepper=stepper, grids=grids)

    # show all
    plotter.show()
