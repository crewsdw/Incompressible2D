import numpy as np
import cupy as cp


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr, axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, orders):
        self.resolutions = resolutions
        self.orders = orders
        # Permutations
        # self.permutations = [(0, 3, 1, 2),  # For contraction with x nodes
        #                      (0, 1, 2, 3)]  # For contraction with y nodes
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
        # self.grid_axis = np.array([0, 2, 4])
        # self.sub_element_axis = np.array([1, 3, 5])
        self.grid_axis = np.array([1, 3])
        self.sub_element_axis = np.array([2, 4])
        # Numerical flux allocation size arrays
        self.num_flux_sizes = [(2, resolutions[0], 2, resolutions[1], orders[1]),
                               (2, resolutions[0], orders[0], resolutions[1], 2)]

    def semi_discrete_rhs(self, vector, elliptic, basis, grids):
        """
        Calculate the right-hand side of semi-discrete equation
        """
        return ((self.x_flux(vector=vector, basis=basis.basis_x) * grids.x.J) +
                (self.y_flux(vector=vector, basis=basis.basis_y) * grids.y.J) +
                self.source_term(elliptic=elliptic))

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

    def source_term(self, elliptic):
        """
        Add the pressure gradient point-wise as a source term in NS momentum equation
        """
        return -1.0 * elliptic.pressure_gradient.arr
