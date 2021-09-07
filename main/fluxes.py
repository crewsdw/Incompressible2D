import numpy as np
import cupy as cp

import matplotlib.pyplot as plt


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr, axes=([axis], [1])),
                        axes=permutation)


class Spectral:
    def __init__(self, resolutions, orders, nu=1.0e-2):
        # Parameters
        self.resolutions = resolutions
        self.orders = orders
        self.nu = nu  # viscosity value

    def semi_discrete_rhs(self, vector, elliptic, basis, grids):
        """
        Experiment: Compute the semi-discrete right-hand side using a purely spectral method
        """
        # Compute right-hand side
        return (self.nu * vector.laplacian(grids=grids) -
                elliptic.pressure_gradient.arr -
                vector.flux_divergence(grids=grids))


class DGFlux:
    def __init__(self, resolutions, orders, experimental_viscosity=False, nu=1.0e-2):
        # Parameters
        self.resolutions = resolutions
        self.orders = orders
        self.nu = nu  # viscosity value
        # Permutations
        self.permutations = [(0, 1, 4, 2, 3),  # For contraction with x nodes
                             (0, 1, 2, 3, 4)]  # For contraction with y nodes
        # Boundary slices
        self.boundary_slices = [
            # x-directed face slices [(comps), (left), (right)]
            [(slice(2), slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1])),
             (slice(2), slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]))],
            # y-directed face slices [(left), (right)]
            [(slice(2), slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0),
             (slice(2), slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1)]]
        # Speed slices [(comps), (left), (right)]
        self.speed_slices = [
            # x-directed face slices [(left), (right)]
            [(slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1])),
             (slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]))],
            # y-directed face slices [(left), (right)]
            [(slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0),
             (slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1)]]

        # Grid and sub-element axes
        self.grid_axis = np.array([1, 3])
        self.sub_element_axis = np.array([2, 4])
        # Numerical flux allocation size arrays
        self.num_flux_sizes = [(2, resolutions[0], 2, resolutions[1], orders[1]),
                               (2, resolutions[0], orders[0], resolutions[1], 2)]

        # Flags
        self.experimental_viscosity = experimental_viscosity

    def semi_discrete_rhs(self, vector, elliptic, basis, grids):
        """
        Calculate the right-hand side of semi-discrete equation
        """
        # X, Y = np.meshgrid(grids.x.arr[1:-1, :].flatten(),
        #                    grids.y.arr[1:-1, :].flatten(), indexing='ij')
        # x_flux = ((self.x_flux(vector=vector, basis=basis.basis_x) * grids.x.J) +
        #           (self.y_flux(vector=vector, basis=basis.basis_y) * grids.y.J))[0, 1:-1, :, 1:-1, :]
        # y_flux = ((self.x_flux(vector=vector, basis=basis.basis_x) * grids.x.J) +
        #           (self.y_flux(vector=vector, basis=basis.basis_y) * grids.y.J))[1, 1:-1, :, 1:-1, :]
        # x_flux_flat = x_flux.reshape((x_flux.shape[0] * x_flux.shape[1], x_flux.shape[2] * x_flux.shape[3]))
        # y_flux_flat = y_flux.reshape((x_flux.shape[0] * x_flux.shape[1], x_flux.shape[2] * x_flux.shape[3]))
        # plt.figure()
        # plt.contourf(X, Y, x_flux_flat.get())
        # plt.colorbar()
        # plt.figure()
        # plt.contourf(X, Y, y_flux_flat.get())
        # plt.colorbar()
        # plt.show()
        # quit()

        return ((self.x_flux(vector=vector, basis=basis.basis_x) * grids.x.J) +
                (self.y_flux(vector=vector, basis=basis.basis_y) * grids.y.J) +
                self.source_term(elliptic=elliptic, vector=vector, grids=grids))

    def x_flux(self, vector, basis):  # , elliptic, grid_x):
        dim = 0
        # Advection: flux is the tensor v_i * v_j
        flux = vector.arr[0, :, :, :, :] * vector.arr[:, :, :, :, :]
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.spatial_flux(flux=flux, speed=vector.arr, basis=basis, dim=dim))

    def y_flux(self, vector, basis):  # , elliptic, grid_y):
        dim = 1
        # Advection: flux is the tensor v_i * v_j
        flux = vector.arr[1, :, :, :, :] * vector.arr[:, :, :, :, :]
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.spatial_flux(flux=flux, speed=vector.arr, basis=basis, dim=dim))

    def spatial_flux(self, flux, speed, basis, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])

        # Measure upwind directions
        speed_neg = cp.where(condition=speed[dim, :, :, :, :] < 0, x=1, y=0)
        speed_pos = cp.where(condition=speed[dim, :, :, :, :] >= 0, x=1, y=0)

        # Upwind flux, left and right faces
        # print(num_flux[self.boundary_slices[dim][0]].shape)
        # print(flux[self.boundary_slices[dim][1]].shape)
        # print(speed_pos[self.speed_slices[dim][0]].shape)
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1, axis=self.grid_axis[dim]),
                                                                     speed_pos[self.speed_slices[dim][0]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     speed_neg[self.speed_slices[dim][0]]))
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              speed_pos[self.speed_slices[dim][1]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              speed_neg[self.speed_slices[dim][1]]))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])

    def source_term(self, elliptic, vector, grids):
        """
        Add source terms in NS momentum equation point-wise: the pressure gradient and experimental_viscosity
        """
        if self.experimental_viscosity:
            return self.nu * vector.laplacian(grids=grids) - elliptic.pressure_gradient.arr
        else:
            return -1.0 * elliptic.pressure_gradient.arr

# Temp bin

# v_dot_grad_v_spectrum = (cp.multiply(1j * grids.x.d_wave_numbers[None, :, None], vector.dyad_spectrum[:, 0, :, :]) +
#                          cp.multiply(1j * grids.y.d_wave_numbers[None, None, :], vector.dyad_spectrum[:, 1, :, :]))
# print(v_dot_grad_v_spectrum.shape)
# flux_x = grids.inverse_transform(spectrum=v_dot_grad_v_spectrum[0, :, :])
# flux_y = grids.inverse_transform(spectrum=v_dot_grad_v_spectrum[1, :, :])
# flux_x_grid = flux_x.reshape((flux_x.shape[0] * flux_x.shape[1], flux_x.shape[2] * flux_x.shape[3]))
# flux_y_grid = flux_y.reshape((flux_x.shape[0] * flux_x.shape[1], flux_x.shape[2] * flux_x.shape[3]))

# X, Y = np.meshgrid(grids.x.arr[1:-1, :].flatten(),
#                    grids.y.arr[1:-1, :].flatten(), indexing='ij')
#
# plt.figure()
# plt.contourf(X, Y, flux_x_grid.get())
# plt.colorbar()
# plt.figure()
# plt.contourf(X, Y, flux_y_grid.get())
# plt.colorbar()
# plt.show()
# quit()