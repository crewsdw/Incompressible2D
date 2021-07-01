import numpy as np
import cupy as cp
import scipy.special as sp

# For debug
import matplotlib.pyplot as plt


# noinspection PyTypeChecker
class Grid1D:
    def __init__(self, low, high, res, basis, spectrum=False, fine=False):
        self.low = low
        self.high = high
        self.res = int(res)  # somehow gets non-int...
        self.res_ghosts = int(res + 2)  # resolution including ghosts
        self.order = basis.order

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.res

        # element Jacobian
        self.J = 2.0 / self.dx

        # The grid does not have a basis but does have quad weights
        self.quad_weights = cp.tensordot(cp.ones(self.res), cp.asarray(basis.weights), axes=0)
        # arrays
        self.arr = np.zeros((self.res_ghosts, self.order))
        self.create_grid(basis.nodes)
        self.arr_cp = cp.asarray(self.arr)
        self.midpoints = np.array([(self.arr[i, -1] + self.arr[i, 0]) / 2.0 for i in range(1, self.res_ghosts - 1)])
        self.arr_max = np.amax(abs(self.arr))

        # velocity axis gets a positive/negative indexing slice
        self.one_negatives = cp.where(condition=self.arr_cp < 0, x=1, y=0)
        self.one_positives = cp.where(condition=self.arr_cp >= 0, x=1, y=0)

        # fine array
        if fine:
            fine_num = 25  # 200 for 1D poisson study
            self.arr_fine = np.array([np.linspace(self.arr[i, 0], self.arr[i, -1], num=fine_num)
                                      for i in range(self.res_ghosts)])

        # spectral coefficients
        if spectrum:
            self.nyquist_number = 2 * self.length // self.dx  # 2.5 *  # mode number of nyquist frequency
            # print(self.nyquist_number)
            self.k1 = 2.0 * np.pi / self.length  # fundamental mode
            self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
            self.d_wave_numbers = cp.asarray(self.wave_numbers)
            self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))

            # Spectral matrices
            self.spectral_transform = basis.fourier_transform_array(self.midpoints, self.J, self.wave_numbers)
            self.inverse_transform = basis.inverse_transform_array(self.midpoints, self.J, self.wave_numbers)

    def create_grid(self, nodes):
        """
        Initialize array of global coordinates (including ghost elements).
        """
        # shift to include ghost cells
        min_gs = self.low - self.dx
        max_gs = self.high  # + self.dx
        # nodes (iso-parametric)
        nodes = (np.array(nodes) + 1) / 2

        # element left boundaries (including ghost elements)
        xl = np.linspace(min_gs, max_gs, num=self.res_ghosts)

        # construct coordinates
        for i in range(self.res_ghosts):
            self.arr[i, :] = xl[i] + self.dx * nodes

    def grid2cp(self):
        self.arr = cp.asarray(self.arr)

    def grid2np(self):
        self.arr = self.arr.get()

    def fourier_basis(self, function, idx):
        """
        On GPU, compute Fourier coefficients on the LGL grid of the given grid function
        """
        # print(function.shape)
        # print(self.spectral_transform.shape)
        return cp.tensordot(function, self.spectral_transform, axes=(idx, [0, 1])) * self.dx / self.length

    def sum_fourier(self, coefficients, idx):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        # return cp.real(cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0])))
        # print(cp.where(cp.isnan(coefficients) == True))
        # print(coefficients[39])
        # quit()
        # print(self.grid_phases)
        # quit()
        # print(cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0])))
        # print(cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0])))
        # quit()
        return cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0]))

    def inverse_transformation(self, coefficients, idx):
        """
        Experimental: invert Fourier transformation
        """
        return cp.real(cp.tensordot(self.inverse_transform, coefficients, axes=([2], idx)))


class Grid2D:
    def __init__(self, basis, lows, highs, resolutions, fine_all=False):
        # Grids
        self.x = Grid1D(low=lows[0], high=highs[0], res=resolutions[0],
                        basis=basis.basis_x, spectrum=True, fine=fine_all)
        self.y = Grid1D(low=lows[1], high=highs[1], res=resolutions[1],
                        basis=basis.basis_y, spectrum=True, fine=fine_all)
        # res
        self.res_ghosts = [self.x.res_ghosts, self.y.res_ghosts]
        self.orders = [self.x.order, self.y.order]

    def fourier_transform(self, function):
        # Transform function on a 2D grid
        x_transform = cp.transpose(self.x.fourier_basis(function=function, idx=[0, 1]),
                                   axes=(2, 0, 1))
        xy_transform = self.y.fourier_basis(function=x_transform, idx=[1, 2])
        return xy_transform

    def inverse_transform(self, spectrum):
        # Inverse transform spectrum back to piecewise grid
        y_transform = self.y.sum_fourier(coefficients=spectrum, idx=[1])
        xy_transform = self.x.sum_fourier(coefficients=y_transform, idx=[0])
        return cp.real(cp.transpose(xy_transform, axes=(2, 3, 0, 1)))


class Scalar:
    def __init__(self, resolutions, orders, perturbation=True):
        # if perturbation
        self.perturbation = perturbation

        # resolutions (no ghosts)
        self.x_res, self.y_res = resolutions[0], resolutions[1]

        # orders
        self.x_ord, self.y_ord = int(orders[0]), int(orders[1])

        # array
        self.arr = None

        # sizes
        size0 = slice(resolutions[0] + 2)
        size1 = slice(resolutions[1] + 2)

        self.boundary_slices = [
            # x-directed face slices [(left), (right)]
            [(size0, 0, size1, slice(orders[1])),
             (size0, -1, size1, slice(orders[1]))],
            [(size0, slice(orders[0]), size1, 0),
             (size0, slice(orders[0]), size1, -1)]]
        # Grid and sub-element axes
        self.grid_axis = np.array([0, 2])
        self.sub_element_axis = np.array([1, 3])

    def initialize(self, grids):
        # Just sine product...
        x2 = cp.tensordot(grids.x.arr_cp, cp.ones((self.y_res, self.y_ord)), axes=0)
        y2 = cp.tensordot(cp.ones((self.x_res, self.x_ord)), grids.y.arr_cp, axes=0)
        # random function
        self.arr = cp.sin(x2) * cp.sin(y2) * cp.sin(3.0 * x2 + 3.0 * y2) * cp.cos(4.0 * x2 - 5.0 * y2)

    def grid_flatten_gpu(self):
        return self.arr.reshape((self.x_res * self.x_ord, self.y_res * self.y_ord))


class Vector:
    def __init__(self, resolutions, orders, perturbation=True):
        # if perturbation
        self.perturbation = perturbation

        # resolutions
        self.x_res, self.y_res = resolutions

        # orders
        self.x_ord, self.y_ord = int(orders[0]), int(orders[1])

        # arrays
        self.arr = None
        self.arr_stages = None
        self.grad = None
        self.pressure_source = None

        # no ghost slice
        self.no_ghost_slice = (slice(2),
                               slice(1, self.x_res - 1), slice(self.x_ord),
                               slice(1, self.y_res - 1), slice(self.y_ord))

    def initialize(self, grids):
        # Just sine product...
        x2 = cp.tensordot(grids.x.arr_cp, cp.ones((self.y_res, self.y_ord)), axes=0)
        y2 = cp.tensordot(cp.ones((self.x_res, self.x_ord)), grids.y.arr_cp, axes=0)

        # 2D ABC flow
        arr_x = cp.cos(y2) + cp.cos(2.0 * y2) + cp.cos(3.0 * y2)  # + y2) + cp.sin(2.0 * x2)
        arr_y = cp.sin(x2) + cp.sin(2.0 * x2) + cp.sin(3.0 * x2)  # - y2) + cp.cos(2.0 * y2)
        # arr_x = cp.cos(x2)*cp.sin(y2) - cp.cos(3.0*x2)*cp.sin(3.0*y2)
        # arr_y = -cp.sin(x2)*cp.cos(y2) + cp.sin(3.0*x2)*cp.cos(3.0*y2)
        # arr_x = cp.cos(y2) + cp.cos(x2)*cp.sin(y2)
        # arr_y = cp.sin(x2) - cp.sin(x2)*cp.cos(y2)

        self.arr = cp.array([arr_x, arr_y])

    def gradient_tensor(self, grids):
        """
        Compute gradient tensor using spectral method
        """
        # Compute spectrum
        spectrum_x = grids.fourier_transform(function=self.arr[0, 1:-1, :, 1:-1, :])
        spectrum_y = grids.fourier_transform(function=self.arr[1, 1:-1, :, 1:-1, :])

        # Compute spectral derivatives
        dx_fx_k = cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_x)
        dy_fx_k = cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_x)
        dx_fy_k = cp.multiply(1j * grids.x.d_wave_numbers[:, None], spectrum_y)
        dy_fy_k = cp.multiply(1j * grids.y.d_wave_numbers[None, :], spectrum_y)

        # Inverse transform ( dx_fx == d(f_x) / dx )
        dx_fx = grids.inverse_transform(spectrum=dx_fx_k)
        dy_fx = grids.inverse_transform(spectrum=dy_fx_k)
        dx_fy = grids.inverse_transform(spectrum=dx_fy_k)
        dy_fy = grids.inverse_transform(spectrum=dy_fy_k)

        self.grad = cp.array([[dx_fx, dy_fx], [dx_fy, dy_fy]])

    def poisson_source(self):
        """
        Compute double contraction of velocity-gradient tensor
        """
        self.pressure_source = cp.einsum('ijklnm,jiklnm->klnm', self.grad, self.grad)

    def grid_flatten_arr(self):
        return self.arr.reshape((2, self.x_res * self.x_ord, self.y_res * self.y_ord))

    def grid_flatten_grad(self):
        return self.grad.reshape((2, 2, (self.x_res - 2) * self.x_ord, (self.y_res - 2) * self.y_ord))

    def grid_flatten_source(self):
        return self.pressure_source.reshape((self.x_res - 2) * self.x_ord, (self.y_res - 2) * self.y_ord)

    def ghost_sync(self):
        self.arr[:, 0, :, :, :] = self.arr[:, -2, :, :, :]
        self.arr[:, -1, :, :, :] = self.arr[:, 1, :, :, :]
        self.arr[:, :, :, 0, :] = self.arr[:, :, :, -2, :]
        self.arr[:, :, :, -1, :] = self.arr[:, :, :, 1, :]
