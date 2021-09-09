import numpy as np
import cupy as cp

# Set random state
np.random.seed(126)


# For debug
# import matplotlib.pyplot as plt


# noinspection PyTypeChecker
class Grid1D:
    def __init__(self, low, high, res, basis, spectrum=False, fine=False, linspace=False):
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

        if linspace:
            lin_num = self.res * self.order
            self.arr_lin = np.linspace(self.low, self.high, num=lin_num)

        # spectral coefficients
        if spectrum:
            self.nyquist_number = 2.0 * self.res + 1.0  # 2.0 * self.length // self.dx# mode number of nyquist frequency
            self.k1 = 2.0 * np.pi / self.length  # fundamental mode
            self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
            self.d_wave_numbers = cp.asarray(self.wave_numbers)
            self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))

            if linspace:
                self.lin_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr_lin, axes=0)))

            # Spectral matrices
            self.spectral_transform = (self.dx / (2.0 * self.length) *
                                      basis.fourier_transform_array(self.midpoints, self.J, self.wave_numbers))
            # self.inverse_transform = basis.inverse_transform_array(self.midpoints, self.J, self.wave_numbers)

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
        # quit()
        return cp.tensordot(function, self.spectral_transform, axes=(idx, [0, 1]))

    def sum_fourier(self, coefficients, idx):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        return cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0]))

    def sum_fourier_to_linspace(self, coefficients, idx):
        return cp.tensordot(coefficients, self.lin_phases, axes=(idx, [0]))


class Grid2D:
    def __init__(self, basis, lows, highs, resolutions, fine_all=False, linspace=False):
        # Grids
        self.x = Grid1D(low=lows[0], high=highs[0], res=resolutions[0],
                        basis=basis.basis_x, spectrum=True, fine=fine_all, linspace=linspace)
        self.y = Grid1D(low=lows[1], high=highs[1], res=resolutions[1],
                        basis=basis.basis_y, spectrum=True, fine=fine_all, linspace=linspace)
        # res
        self.res_ghosts = [self.x.res_ghosts, self.y.res_ghosts]
        self.orders = [self.x.order, self.y.order]

        # spectral radius squared (for laplacian)
        self.kr_sq = (outer2(self.x.d_wave_numbers, cp.ones_like(self.y.d_wave_numbers)) ** 2.0 +
                      outer2(cp.ones_like(self.x.d_wave_numbers), self.y.d_wave_numbers) ** 2.0)

        # wave number vector
        self.wave_numbers = cp.array([self.x.d_wave_numbers, self.y.d_wave_numbers])

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

    def inverse_transform_linspace(self, spectrum):
        # Inverse transform spectrum back to a linearly spaced grid
        y_transform = self.y.sum_fourier_to_linspace(coefficients=spectrum, idx=[1])
        xy_transform = self.x.sum_fourier_to_linspace(coefficients=y_transform, idx=[0])
        return cp.real(cp.transpose(xy_transform, axes=(1, 0)))

    def laplacian(self, function):
        return self.inverse_transform(spectrum=cp.multiply(-self.kr_sq, self.fourier_transform(function=function)))


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

    def grid_flatten_arr(self):
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
        self.dyad_spectrum = None
        self.pressure_source = None

        # no ghost slice
        self.no_ghost_slice = (slice(2),
                               slice(1, self.x_res - 1), slice(self.x_ord),
                               slice(1, self.y_res - 1), slice(self.y_ord))

    def initialize(self, grids, numbers=None):
        # default argument
        if numbers is None:
            numbers = [2, 3, 4]
        # temp grid variables
        x2 = cp.tensordot(grids.x.arr_cp, cp.ones((self.y_res, self.y_ord)), axes=0)
        y2 = cp.tensordot(cp.ones((self.x_res, self.x_ord)), grids.y.arr_cp, axes=0)

        # 2D ABC flow superposition
        p = np.pi * np.random.randn(len(numbers))  # phases
        arr_x = sum([cp.cos(number * y2 + p[idx]) for idx, number in enumerate(numbers)])
        arr_y = sum([cp.sin(number * x2 + p[idx]) for idx, number in enumerate(numbers)])

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
        self.pressure_source = -1.0 * cp.einsum('ijklnm,jiklnm->klnm', self.grad, self.grad)

    def dyad_transform(self, grids):
        """
        Experimental: compute spectrum of the dyad v_i * v_j
        """
        k_xx = grids.fourier_transform(function=cp.multiply(self.arr[0, 1:-1, :, 1:-1, :],
                                                            self.arr[0, 1:-1, :, 1:-1, :]))
        k_xy = grids.fourier_transform(function=cp.multiply(self.arr[0, 1:-1, :, 1:-1, :],
                                                            self.arr[1, 1:-1, :, 1:-1, :]))
        k_yy = grids.fourier_transform(function=cp.multiply(self.arr[1, 1:-1, :, 1:-1, :],
                                                            self.arr[1, 1:-1, :, 1:-1, :]))

        self.dyad_spectrum = cp.array([[k_xx, k_xy], [k_xy, k_yy]])

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

    def filter(self, grids):
        # Compute spectrum
        spectrum_x = grids.fourier_transform(function=self.arr[0, 1:-1, :, 1:-1, :])
        spectrum_y = grids.fourier_transform(function=self.arr[1, 1:-1, :, 1:-1, :])
        # Inverse transform
        self.arr[0, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=spectrum_x)
        self.arr[1, 1:-1, :, 1:-1, :] = grids.inverse_transform(spectrum=spectrum_y)

    def laplacian(self, grids):
        """
        Return the vector laplacian on the given grids
        :param grids: Grids2D object
        :return: vector Laplacian (u_xx + u_yy, v_xx + v_yy) of size (2, Nx, n, Ny, n)
        """
        laplacian = cp.zeros_like(self.arr)
        laplacian[:, 1:-1, :, 1:-1, :] = cp.array([grids.laplacian(function=self.arr[0, 1:-1, :, 1:-1, :]),
                                                   grids.laplacian(function=self.arr[1, 1:-1, :, 1:-1, :])])
        return laplacian

    def flux_divergence(self, grids):
        """
        Return the vector laplacian on the given grids
        :param grids: Grids2D object
        :return: flux divergence d_i * (v_i * v_j) of size (2, Nx, n, Ny, n)
        """
        # Allocate
        flux_divergence = cp.zeros_like(self.arr)
        # Compute transform of flux tensor (v_i * v_j)
        self.dyad_transform(grids=grids)
        # Compute spectral divergence of flux tensor
        k_flux_divergence = (
                cp.multiply(1j * grids.x.d_wave_numbers[None, :, None], self.dyad_spectrum[:, 0, :, :]) +
                cp.multiply(1j * grids.y.d_wave_numbers[None, None, :], self.dyad_spectrum[:, 1, :, :])
        )
        # Inverse transform
        flux_divergence[:, 1:-1, :, 1:-1, :] = cp.array([grids.inverse_transform(spectrum=k_flux_divergence[0, :, :]),
                                                         grids.inverse_transform(spectrum=k_flux_divergence[1, :, :])])
        return flux_divergence


def outer2(a, b):
    """
    Compute outer tensor product of vectors a, b
    :param a: vector a_i
    :param b: vector b_j
    :return: tensor a_i b_j
    """
    return cp.tensordot(a, b, axes=0)

#
# def inv_rot_x(x2, y2, angle):
#     rot_x2 = cp.cos(angle) * x2 + cp.sin(angle) * y2
#     return rot_x2
#
#
# def inv_rot_y(x2, y2, angle):
#     rot_y2 = -cp.sin(angle) * x2 + cp.cos(angle) * y2
#     return rot_y2

# Trash bin
# q = np.pi * np.random.randn(number)
# arr_x = cp.cos(y2 + p1) + cp.cos(2.0 * y2 + p2) + cp.cos(3.0 * y2 + p3) +  # + y2) + cp.sin(2.0 * x2)
# arr_y = cp.sin(x2 + p1) + cp.sin(2.0 * x2 + p2) + cp.sin(3.0 * x2 + p3)  # - y2) + cp.cos(2.0 * y2)
# rot_x2, rot_y2 = np.zeros_like(q), np.zeros_like(q)
# for idx, angle in enumerate(q):
#     rot_x2[idx] = cp.cos(angle) * x2 + cp.sin(angle) * y2
#     rot_y2[idx] = -cp.sin(angle) * x2 + cp.cos(angle) * y2
# arr_x = sum([cp.cos((i+1) * inv_rot_y(x2, y2, q[i]) + p[i]) * cp.cos(q[i]) -
#              cp.sin((i+1) * inv_rot_x(x2, y2, q[i]) + p[i]) * cp.sin(q[i])
#              for i in range(number)])
# arr_y = sum([cp.cos((i+1) * inv_rot_y(x2, y2, q[i]) + p[i]) * cp.sin(q[i]) +
#              cp.sin((i+1) * inv_rot_x(x2, y2, q[i]) + p[i]) * cp.cos(q[i])
#              for i in range(number)])
