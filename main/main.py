import numpy as np
import cupy as cp
import grid as g
import basis as b
import elliptic as ell
import fluxes as fx
import timestep as ts

import matplotlib.pyplot as plt

# Parameters
order = 8
res_x, res_y = 50, 50

# Flags
plot_IC = True

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
grids = g.Grid2D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)

# Time info
final_time = 2.0 * np.pi

# Initialize variable
source = g.Scalar(resolutions=resolutions_ghosts, orders=orders)
source.initialize(grids=grids)

# Visualize
X, Y = np.meshgrid(grids.x.arr[:, :].flatten(), grids.y.arr[:, :].flatten(), indexing='ij')

# ng = [slice(1, -1), slice(None), slice(1, -1), slice(None)]
ng = (slice(order, -order), slice(order, -order))
ng0 = (0, slice(order, -order), slice(order, -order))
ng1 = (1, slice(order, -order), slice(order, -order))

nt00 = (0, 0, slice(None), slice(None))  # slice(order, -order), slice(order, -order))
nt01 = (0, 1, slice(None), slice(None))  # slice(order, -order), slice(order, -order))
nt10 = (1, 0, slice(None), slice(None))  # slice(order, -order), slice(order, -order))
nt11 = (1, 1, slice(None), slice(None))  # slice(order, -order), slice(order, -order))

# if plot_IC:
#     plt.figure()
#     cb = np.linspace(cp.amin(source.arr), cp.amax(source.arr), num=100).get()
#     plt.contourf(X[ng], Y[ng], source.grid_flatten_gpu().get()[ng], cb)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.colorbar()
#
#     plt.show()

# Test vector-valued variable
velocity = g.Vector(resolutions=resolutions_ghosts, orders=orders)
velocity.initialize(grids=grids)

if plot_IC:
    plt.figure()
    cb = np.linspace(cp.amin(velocity.arr), cp.amax(velocity.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], velocity.grid_flatten_arr().get()[ng0], cb)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('x-component')
    plt.colorbar()

    plt.figure()
    cb = np.linspace(cp.amin(velocity.arr), cp.amax(velocity.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], velocity.grid_flatten_arr().get()[ng1], cb)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y-component')
    plt.colorbar()

    plt.show()

# Test elliptic class and pressure solve
elliptic = ell.Elliptic(grids=grids)
elliptic.pressure_solve(velocity=velocity, grids=grids)

# Check it out
plt.figure()
cb = np.linspace(cp.amin(elliptic.pressure.arr), cp.amax(elliptic.pressure.arr), num=100).get()
plt.contourf(X[ng], Y[ng], elliptic.pressure.grid_flatten_gpu().get()[ng], cb)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Try solution to some time
dg_flux = fx.DGFlux(resolutions=resolutions_ghosts, orders=orders)
stepper = ts.Stepper(time_order=3, space_order=order, write_time=0.1, final_time=final_time)

print('\nBeginning main loop...')
stepper.main_loop(vector=velocity, basis=basis, elliptic=elliptic, grids=grids, dg_flux=dg_flux)

print('\nVisualizing final state')
if plot_IC:
    plt.figure()
    cb = np.linspace(cp.amin(velocity.arr), cp.amax(velocity.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], velocity.grid_flatten_arr().get()[ng0], cb)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('x-component')
    plt.colorbar()

    plt.figure()
    cb = np.linspace(cp.amin(velocity.arr), cp.amax(velocity.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], velocity.grid_flatten_arr().get()[ng1], cb)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y-component')
    plt.colorbar()

    plt.figure()
    cb = np.linspace(cp.amin(elliptic.pressure.arr), cp.amax(elliptic.pressure.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], elliptic.pressure.grid_flatten_gpu().get()[ng], cb)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pressure')

    plt.show()

# Bin
# # Velocity-gradient tensor
# velocity.gradient_tensor(grids=grids)
# velocity.poisson_source()
#
# if plot_IC:
#     plt.figure()
#     cb = np.linspace(cp.amin(velocity.grad), cp.amax(velocity.grad), num=100).get()
#     plt.contourf(X[ng], Y[ng], velocity.grid_flatten_grad().get()[nt00], cb)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('xx-component')
#     plt.colorbar()
#
#     plt.figure()
#     cb = np.linspace(cp.amin(velocity.grad), cp.amax(velocity.grad), num=100).get()
#     plt.contourf(X[ng], Y[ng], velocity.grid_flatten_grad().get()[nt01], cb)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('xy-component')
#     plt.colorbar()
#
#     plt.figure()
#     cb = np.linspace(cp.amin(velocity.grad), cp.amax(velocity.grad), num=100).get()
#     plt.contourf(X[ng], Y[ng], velocity.grid_flatten_grad().get()[nt10], cb)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('yx-component')
#     plt.colorbar()
#
#     plt.figure()
#     cb = np.linspace(cp.amin(velocity.grad), cp.amax(velocity.grad), num=100).get()
#     plt.contourf(X[ng], Y[ng], velocity.grid_flatten_grad().get()[nt11], cb)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('yy-component')
#     plt.colorbar()
#
#     plt.figure()
#     cb = np.linspace(cp.amin(velocity.pressure_source), cp.amax(velocity.pressure_source), num=100).get()
#     plt.contourf(X[ng], Y[ng], velocity.grid_flatten_source().get(), cb)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('pressure poisson source')
#     plt.colorbar()
#
#     plt.show()

# Fourier transform pressure source
# spectrum = grids.fourier_transform(function=velocity.pressure_source)
#
# KX, KY = np.meshgrid(grids.x.wave_numbers, grids.y.wave_numbers, indexing='ij')
#
# plt.figure()
# cb = np.linspace(cp.amin(cp.real(spectrum)), cp.amax(cp.real(spectrum)), num=100).get()
# plt.contourf(KX, KY, np.real(spectrum.get()), cb)
# plt.xlabel('kx')
# plt.ylabel('ky')
#
# plt.figure()
# cb = np.linspace(cp.amin(cp.imag(spectrum)), cp.amax(cp.imag(spectrum)), num=100).get()
# plt.contourf(KX, KY, np.imag(spectrum.get()), cb)
# plt.xlabel('kx')
# plt.ylabel('ky')
# plt.show()
#
# # Solve Poisson
# ikx = cp.tensordot(grids.x.d_wave_numbers, cp.ones_like(grids.y.wave_numbers), axes=0)
# iky = cp.tensordot(cp.ones_like(grids.x.wave_numbers), grids.y.d_wave_numbers, axes=0)
#
# poisson_spectrum = -1.0 * cp.divide(spectrum, ikx ** 2.0 + iky ** 2.0)
# poisson_spectrum = cp.nan_to_num(poisson_spectrum)
#
# # Resum spectrum
# potential = g.Scalar(resolutions=resolutions_ghosts, orders=orders)
# potential.arr = cp.zeros_like(source.arr)
# potential.arr[1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=poisson_spectrum)

# Experiment with 1D inversion
# y = cp.sin(grids.x.arr_cp[1:-1, :]) # + cp.cos(2.0 * grids.x.arr_cp[1:-1, :]) + cp.sin(10.0 * grids.x.arr_cp[1:-1, :])
# spectrum_1d = grids.x.fourier_basis(function=y, idx=[0, 1])
#
# # Inverse transform array
# inverse = grids.x.inverse_transformation(coefficients=spectrum_1d, idx=[0])
#
# plt.figure()
# plt.semilogy(grids.x.wave_numbers, np.absolute(spectrum_1d.get()), 'o--')
# plt.xlabel('k')
# plt.ylabel(r'spectrum $|c(k)|$')
#
# plt.figure()
# plt.plot(grids.x.arr[1:-1, :].flatten(), y.get().flatten(), 'o--')
# plt.plot(grids.x.arr[1:-1, :].flatten(), np.real(inverse.get().flatten()), 'o--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# print(spectrum_1d.shape)
#
# quit()
