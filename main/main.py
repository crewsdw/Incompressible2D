import numpy as np
import cupy as cp
import grid as g
import basis as b
import elliptic as ell
import fluxes as fx
import timestep as ts
import matplotlib.animation as animation
# from scipy.interpolate import interp2d

import matplotlib.pyplot as plt

# Parameters
order = 8
res_x, res_y = 30, 30

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
grids = g.Grid2D(basis=basis, lows=lows, highs=highs, resolutions=resolutions, linspace=True)

# Time info
final_time = 0.5  # 13.0  # 10.0 * np.pi
write_time = 0.1

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

    # Interpolate 2D
    XE = np.tensordot(grids.x.arr_lin, np.ones_like(grids.y.arr_lin), axes=0)
    YE = np.tensordot(np.ones_like(grids.x.arr_lin), grids.y.arr_lin, axes=0)
    # fourier interpolation
    spectrum_x = grids.fourier_transform(function=velocity.arr[0, 1:-1, :, 1:-1, :])
    spectrum_y = grids.fourier_transform(function=velocity.arr[1, 1:-1, :, 1:-1, :])
    UE = grids.inverse_transform_linspace(spectrum=spectrum_x)
    VE = grids.inverse_transform_linspace(spectrum=spectrum_y)
    V = np.sqrt(UE.get() ** 2.0 + VE.get() ** 2.0).transpose()

    start_points = np.array([np.random.random_sample(300) * L - L / 2,
                             np.random.random_sample(300) * L - L / 2]).transpose()
    # start_points = np.array([np.linspace(-L/2, L/2, num=100),
    #                          np.linspace(-L/2, L/2, num=100)]).transpose()
    plt.figure()
    plt.streamplot(YE, XE,
                   UE.get().transpose(), VE.get().transpose(),
                   density=5.0, start_points=start_points, color=V)
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
stepper = ts.Stepper(time_order=3, space_order=order,
                     write_time=write_time, final_time=final_time)

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
    plt.tight_layout()

    plt.figure()
    cb = np.linspace(cp.amin(velocity.arr), cp.amax(velocity.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], velocity.grid_flatten_arr().get()[ng1], cb)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y-component')
    plt.colorbar()
    plt.tight_layout()

    plt.figure()
    cb = np.linspace(cp.amin(elliptic.pressure.arr), cp.amax(elliptic.pressure.arr), num=100).get()
    plt.contourf(X[ng], Y[ng], elliptic.pressure.grid_flatten_gpu().get()[ng], cb)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pressure')
    plt.tight_layout()

    # plt.show()

# ax.axis('equal')
# figv, axv = plt.subplots()
#
# maxs = [np.amax(step) for step in stepper.saved_array]
# mins = [np.amin(step) for step in stepper.saved_array]
# all_max = np.amax(maxs)
# all_min = np.amin(mins)
# cb = np.linspace(all_min, all_max, num=100)

#
# def animate_velocity(idx):
#     axv.collections, axv.patches = [], []
#     spectrum_x = grids.fourier_transform(function=cp.asarray(
#         stepper.saved_array[idx][0, 1:-1, :, 1:-1, :]))
#     spectrum_y = grids.fourier_transform(function=cp.asarray(
#         stepper.saved_array[idx][1, 1:-1, :, 1:-1, :]))
#     UE = grids.inverse_transform_linspace(spectrum=spectrum_x)
#     VE = grids.inverse_transform_linspace(spectrum=spectrum_y)
#     V = np.sqrt(UE.get() ** 2.0 + VE.get() ** 2.0)
#
#     axv.contourf(XE, YE, V, cb)
#     axv.set_title('Velocity, t=' + str(stepper.saved_times[idx]))


# Animation of streamlines
fig, ax = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
# plt.tight_layout()
cb = np.linspace(0, np.sqrt(2.0) * np.amax(np.absolute(stepper.saved_array)), num=100)
KX, KY = np.meshgrid(grids.x.wave_numbers, grids.y.wave_numbers, indexing='ij')

# Colorbar
spectrum_x = grids.fourier_transform(function=cp.asarray(
        stepper.saved_array[-1][0, 1:-1, :, 1:-1, :]))
spectrum_y = grids.fourier_transform(function=cp.asarray(
        stepper.saved_array[-1][1, 1:-1, :, 1:-1, :]))
div = grids.inverse_transform_linspace(
        spectrum=(cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_x) +
                  cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_y))).get()
vor = grids.inverse_transform_linspace(
        spectrum=(cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_y) -
                  cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_x))).get()
cb_d = np.linspace(np.amin(div), np.amax(div), num=100)
cb_v = np.linspace(np.amin(vor), np.amax(vor), num=100)
print('Maximum divergence error is {:.2f}'.format(np.amax(np.absolute(cb_d))))


def animate_streamlines(idx):
    ax[0].collections = []
    ax[0].patches = []
    ax[1].collections = []
    ax[1].patches = []

    # Obtain velocity
    spectrum_x = grids.fourier_transform(function=cp.asarray(
        stepper.saved_array[idx][0, 1:-1, :, 1:-1, :]))
    spectrum_y = grids.fourier_transform(function=cp.asarray(
        stepper.saved_array[idx][1, 1:-1, :, 1:-1, :]))
    UE = grids.inverse_transform_linspace(spectrum=spectrum_x)
    VE = grids.inverse_transform_linspace(spectrum=spectrum_y)
    V = cp.sqrt(UE ** 2.0 + VE ** 2.0).get()
    # Obtain spectrum
    # V_fg = np.sqrt(stepper.saved_array[idx][0, 1:-1, :, 1:-1, :] ** 2.0 +
    #                stepper.saved_array[idx][1, 1:-1, :, 1:-1, :] ** 2.0)
    # dV = V_fg - np.mean(V_fg)
    # spectrum_V = np.absolute(grids.fourier_transform(function=cp.asarray(dV)).get())
    # Obtain vorticity
    vorticity = grids.inverse_transform_linspace(
        spectrum=(cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_y) -
                  cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_x))
    ).get()

    # Obtain divergence
    # divergence = grids.inverse_transform_linspace(
    #     spectrum=(cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_x) +
    #               cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_y))
    # ).get()

    # Plot fluid momentum
    m_idx = 0
    ax[m_idx].set_xlim(-L / 2, L / 2)
    ax[m_idx].set_ylim(-L / 2, L / 2)
    ax[m_idx].contourf(XE, YE, V, cb)
    ax[m_idx].set_title(r'Fluid momentum $|v|(x,y)$')

    # Plot momentum spectrum
    # ax[1].set_xlim(-20, 20)
    # ax[1].set_ylim(-20, 20)
    # ax[1].contourf(KX, KY, spectrum_V, levels=np.linspace(0, np.amax(spectrum_V), num=100))
    # ax[1].set_title(r'Spectrum $|v|(kx,ky)$')

    # Plot vorticity
    v_idx = 1
    ax[v_idx].set_xlim(-L / 2, L / 2)
    ax[v_idx].set_ylim(-L / 2, L / 2)
    ax[v_idx].contourf(XE, YE, vorticity, cb_v)
    ax[v_idx].set_title(r'Fluid vorticity $\zeta(x,y)$')
    # fig.colorbar(cfv, ax=ax[v_idx])
    fig.suptitle('Time t=' + str(stepper.saved_times[idx]))

    # Plot divergence
    # d_idx = 1
    # ax[d_idx].set_xlim(-L / 2, L / 2)
    # ax[d_idx].set_ylim(-L / 2, L / 2)
    # ax[d_idx].contourf(XE, YE, divergence, cb_d)
    # ax[d_idx].set_title(r'Momentum divergence $\nabla\cdot v(x,y)$')
    # fig.colorbar(cfd, ax=ax[d_idx])

    # Figure super title
    fig.suptitle('Time t={:.2f}'.format(stepper.saved_times[idx]))

    # ax.streamplot(YE, XE,
    #               UE.get().transpose(), VE.get().transpose(),
    #               density=2.0, start_points=start_points, color=V)
    # ax.set_title('Streamlines, t=' + str(stepper.saved_times[idx]))
    # plt.show()
    # print('finishing interpolation')


anim_str = animation.FuncAnimation(fig, animate_streamlines, frames=len(stepper.saved_array))
anim_str.save(filename='..\\movies\\animation.mp4')
# anim_vel = animation.FuncAnimation(fig, animate_velocity, frames=len(stepper.saved_array))

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
