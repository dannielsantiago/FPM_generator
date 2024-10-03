#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Thu Apr 23 22:19:47 2020

@author: r2d2
"""
import numpy as np
from math import log, log10, pi
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation as register_translation
from scipy.ndimage import shift, gaussian_filter, center_of_mass
from numpy.fft import fft2, fftshift
import scipy.ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


def add_complex_colorwheel(fig, ax, loc=4, width=0.2, height=0.2, pad=0.05):
    """
    Add a complex colorwheel to the given figure and axes.

    Parameters:
    - fig: matplotlib figure object
    - ax: matplotlib axes object
    - loc: Integer (1, 2, 3, or 4) specifying the corner (default: 4)
           1: upper right, 2: upper left, 3: lower left, 4: lower right
    - width: Width of the colorwheel (as a fraction of the main axes)
    - height: Height of the colorwheel (as a fraction of the main axes)
    - complex2rgb: Function to convert complex numbers to RGB values

    Returns:
    - axins: The axes object of the colorwheel
    """

    # Create the colorwheel data
    r = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    r, theta = np.meshgrid(r, theta)
    z = r * np.exp(1j * theta)
    rgb = complex2rgb(z) / 255

    # Create a new axes for the colorwheel
    axins = fig.add_axes([0, 0, 1, 1], projection='polar')

    # Set the position based on loc
    if loc == 1:  # upper right
        ip = InsetPosition(ax, [1 - width + pad, 1 - height+ pad, width, height])
    elif loc == 2:  # upper left
        ip = InsetPosition(ax, [0 + pad, 1 - height + pad, width, height])
    elif loc == 3:  # lower left
        ip = InsetPosition(ax, [0 + pad, 0 + pad, width, height])
    else:  # loc == 4, lower right
        ip = InsetPosition(ax, [1 - width + pad, 0 + pad, width, height])

    axins.set_axes_locator(ip)

    # Turn off the grid explicitly
    axins.grid(False)

    # Plot the colorwheel in the inset axes
    im_wheel = axins.pcolormesh(theta, r, np.ones_like(r), color=rgb.reshape(-1, 3))

    # Remove all ticks and labels
    axins.set_yticks([])
    axins.set_xticks([])

    # Add phase labels inside the circle
    label_radius = 0.75  # Adjust this value to move labels radially
    labels = ['0', 'π/2', 'π', '3π/2']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    for angle, label in zip(angles, labels):
        axins.text(angle, label_radius, label, ha='center', va='center')

    # Remove the circle's outline
    axins.spines['polar'].set_visible(False)

    return axins


def setCustomColorMap():
    """
    create the colormap for diffraction data (the same as matlab)
    return: customized matplotlib colormap
    """
    colors = [
        (0.0, 0.0, 0.2),
        (0, 0.0875, 1),
        (0, 0.4928, 1),
        (0, 1, 0),
        (1, 0.6614, 0),
        (1, 0.4384, 0),
        (0.8361, 0, 0),
        (0.6505, 0, 0),
        (0.4882, 0, 0),
    ]

    n = 255  # Discretizes the interpolation into n bins
    cm = LinearSegmentedColormap.from_list("cmap", colors, n)
    return cm

CMAP_DIFFRACTION = setCustomColorMap()

# plt.rcParams['text.usetex'] = True
# # plt.rcParams['font.size'] = 15
# # plt.rcParams['legend.fontsize'] = 18
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['font.family'] = 'serif'


def ifft2c(array):
    """
    performs 2 - dimensional inverse Fourier transformation, where energy is reserved abs(G)**2==abs(fft2c(g))**2
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G along the last two axes
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array), norm='ortho'))


def fft2c(array):
    """
    performs 2 - dimensional unitary Fourier transformation, where energy is reserved abs(g)**2==abs(fft2c(g))**2
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g along the last two axes
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array), norm='ortho'))


def binning_1d(arr, binFactor):
    # Calculate the length of the new array after binning
    new_length = arr.shape[0] // binFactor

    # Reshape the array to a 2D array where each row has binFactor elements
    reshaped_array = arr[:new_length * binFactor].reshape((new_length, binFactor))

    # Sum along the rows to bin the data
    binned_array = reshaped_array.sum(axis=1)

    return binned_array

def binning(arr, binFactor):
    shape = (arr.shape[0] // binFactor, binFactor,
             arr.shape[1] // binFactor, binFactor)
    return arr.reshape(shape).sum(-1).sum(1)


def bin2(X):
    """
    perform 2-by-2 binning.
    :Params X: input 2D image for binning
    return: Y: output 2D image after 2-by-2 binning
    """
    # simple 2-fold binning
    m, n = X.shape
    Y = np.sum(X.reshape(2, m // 2, 2, n // 2), axis=(0, 2))
    return Y


def generateFermatGrid(n, radius, minStep):
    """
    see https://en.wikipedia.org/wiki/Fermat%27s_spiral
    :param n: number of points generated
    :param radius: radius of spiral in meters
    :return: scanPositions
    """
    # golden ratio
    base = np.append(np.arange(n), 0)
    base = np.arange(n)

    r = np.sqrt(base)
    theta0 = (137.508 / 180) * np.pi
    theta = base * theta0

    Xpos = (r * np.cos(theta))
    Ypos = (r * np.sin(theta))

    scanPos = np.array((Ypos, Xpos)).T
    scanPos *= radius / np.amax(scanPos)

    scanPos = scanPos // minStep
    scanPos *= minStep
    # scanPos = np.around(scanPos*1e6, decimals=2)

    return scanPos


def zero_pad(arr):
    """
    Pad arr with zeros to double the size. Only the last 2 dimensions are affected.
    """
    # Determine the new shape with doubled size in the last two dimensions
    new_shape = arr.shape[:-2] + (arr.shape[-2] * 2, arr.shape[-1] * 2)
    out_arr = np.zeros(new_shape, dtype=arr.dtype)

    # Compute the starting indices for the original array within the padded array
    as1 = (arr.shape[-2] + 1) // 2
    as2 = (arr.shape[-1] + 1) // 2

    # Place the original array in the center of the new zero-padded array
    out_arr[..., as1:as1 + arr.shape[-2], as2:as2 + arr.shape[-1]] = arr
    return out_arr


def zero_unpad(arr, original_shape):
    """
    Strip off padding of arr with zeros to halve the size. Only the last 2 dimensions are affected.
    """
    # Compute the starting indices for the subarray to extract
    as1 = (original_shape[-2] + 1) // 2
    as2 = (original_shape[-1] + 1) // 2

    # Extract the subarray that corresponds to the original array's shape
    return arr[..., as1:as1 + original_shape[-2], as2:as2 + original_shape[-1]]


def generateRectangularGrid(step, minStep, Lx, Ly, noiseP=0.15):
    """
    step: distance in [m] of the scanning step. ie = 20e-6
    minStep: min step distance supported by the XYstage. ie = 5e-6
    Lx: length [m] of the grid in the x-direction. ie = 300e-6
    Ly: lenght [m] of the grid in the y-direction ie= 200e-6
    noiseP: percentage of noise to add to the grid, if None = 0.15 [15%]
    """
    ratio = abs(Lx / Ly)
    Nx = int(Lx / step) + 1
    Ny = Nx + int((Nx - 1) * ((1 / ratio) - 1))
    n = int(Nx * Ny)
    print(f'n:{n}, Nx:{Nx}, Ny:{Ny}, ratio:{ratio}')
    x = np.linspace(-1, 1, Nx) * Lx / 2
    y = np.linspace(-1, 1, Ny) * Ly / 2
    Y, X = np.meshgrid(y, x)
    Y[1::2, :] = np.flip(Y[1::2, :])

    x = np.reshape(X, (1, n))
    y = np.reshape(Y, (1, n))

    scanPos = np.concatenate((y, x)).T
    center = np.expand_dims(scanPos[0, :] * 0, axis=0)
    scanPos = np.concatenate((center, scanPos))

    variation = 1 + np.random.rand(*scanPos.shape) * noiseP
    variation *= step
    scanPos += variation

    scanPos = scanPos // minStep
    scanPos *= minStep
    scanPos -= scanPos[0, :]
    # scanPos = np.around(scanPos * 1e6, decimals=2)

    return scanPos


def circ(x, y, D):
    """
    generate a circle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param y: 2D y coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param D: diameter
    :return: a 2D array
    """
    circle = (x ** 2 + y ** 2) < (D / 2) ** 2
    return circle

def circ_px(N, D):
    """
    generate a circle on a 2D grid
    :param N: lateral size of array in px
    :param D: diameter in px
    :return: a 2D array
    """
    x = np.linspace(-N//2, N//2, N, endpoint=False).reshape(1,N)
    y = x.reshape(N,1)
    circle = (x ** 2 + y ** 2) < (D / 2) ** 2
    return circle

def rect_px(N, D):
    """
    generate a rectngle on a 2D grid
    :param N: lateral size of array in px
    :param D: lateral size of square in px
    :return: a 2D array
    """
    x = np.linspace(-N // 2, N // 2, N, endpoint=False).reshape(1, N)
    y = x.reshape(N, 1)
    square = (x**2 <= (D / 2)**2) * (y**2 <= (D / 2)**2)
    return square


def rect(arr, threshold = 0.5):
    """
    generate a binary array containing a rectangle on a 2D grid
    :param x: 2D x coordinate, normally calculated from meshgrid: x,y = np.meshgird((,))
    :param threshold: threshold value to binarilize the input array, default value 0.5
    :return: a binary array
    """
    arr = abs(arr)
    return arr<threshold

def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb (5 times faster than colorsys).
    https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
    h,s should be a numpy arrays with values between 0.0 and 1.0
    v should be a numpy array with values between 0.0 and 255.0
    :param hsv: np.ndarray of shape (x,y,3)
    :return: hsv2rgb returns an array of uints between 0 and 255.
    """
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def complex2rgb(u, amplitudeScalingFactor=1, scalling=1):
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    # if u is on the GPU, remove it as we can toss it now.
    h = np.angle(u).astype(float)
    h = (h + np.pi) / (2 * np.pi)
    # saturation  (ones)
    s = np.ones_like(h)
    # value (normalize brightness to 8-bit)
    v = np.abs(u)
    if amplitudeScalingFactor != 1:
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    if scalling != 1:
        local_max = np.max(v)
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)
        print(f'ratio: {local_max / scalling}, max(v): {np.max(v)}')

        v *= local_max / scalling
        print(f'max(v): {np.max(v)}')

    else:
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb


def complex2rgb2(u, amplitudeScalingFactor=1, scalling=1):
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    # if u is on the GPU, remove it as we can toss it now.
    h = np.angle(u)
    # plot cos(x)
    h = np.real(np.exp(1j * h))
    h = (h + 1) / (4)
    # plot arcos(x)
    # h = np.arccos(np.cos(h))
    # h /= 2*np.pi
    # saturation  (ones)
    s = np.ones_like(h)
    # value (normalize brightness to 8-bit)
    v = np.abs(u)
    if amplitudeScalingFactor != 1:
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    if scalling != 1:
        local_max = np.max(v)
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)
        print(f'ratio: {local_max / scalling}, max(v): {np.max(v)}')

        v *= local_max / scalling
        print(f'max(v): {np.max(v)}')

    else:
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb


path_avg_overlap = lambda c, d: np.mean(1 - np.array([np.linalg.norm(c[p] - c[p + 1]) for p in range(len(c) - 1)]) / d)


def makeGrating2(gratingFreq, shape=(500, 500), dxp=1e-6, binary=True):
    period = 1 / (dxp * gratingFreq * 1000)  # *1000 mmm/m
    L = shape[-1]
    nLines = int(L / period)
    hole = int(2)
    wall = int(period - hole)

    grating = np.zeros(shape)
    # spacing=shape[-1]//(gratingFreq*2)
    for i in range(nLines + 1):
        a = int(i * period + shape[-1] // 2 - hole // 2)
        b = a + hole
        c = b + wall  # int((i + 1) * period + shape[-1] // 2 + hole)
        grating[:, a:b] = 1
        grating[:, b:c] = 0

    grating[:, shape[-1] // 2:0:-1] = grating[:, shape[-1] // 2::]

    return grating


def get_m1_m2(x, y):
    dx = np.zeros_like(y)

    for i in range(len(x) - 1):
        dx[i] = x[i + 1] - x[i]

    '''calculate mean'''
    m1 = np.sum(x * y * dx) / np.sum(y * dx)
    '''calculate variance'''
    m2 = np.sum(((x - m1) ** 2) * y * dx) / np.sum(y * dx)

    return m1, m2


def get_m1(x, y):
    dx = np.zeros_like(y)

    for i in range(len(x) - 1):
        dx[i] = x[i + 1] - x[i]

    '''calculate mean'''
    m1 = np.sum(x * y * dx) / np.sum(y * dx)

    return m1


def lcoh(w, dw, type='gaussian', a=1):
    lcoh = w ** 2 / dw
    # lcoh = np.outer(w**2,1/dw)

    if type == 'gaussian':  # ~0.4
        a = (2 * log(2) / pi)
    if type == 'lorentzian':  # ~0.62
        a = 2 / pi
    if type == 'other':
        a = a

    return a * lcoh


def find_nearest(array, arrax, value):
    array = np.asarray(array)
    try:
        idxs = np.argwhere(np.logical_and(np.abs(array) > value * 0.95, np.abs(array) < value * 1.05))
        idx = np.amax(idxs)
    except:
        idx = np.amax(np.argwhere(array == np.amin(array)))
        # print(f'lcoh:{arrax[idx]*1e6:.1f} um')
    return idx


def compute_Autocorrelation_and_get_lcoh(l, spectrum):
    # print(l.shape)
    # print(spectrum.shape)
    # l = np.expand_dims(l, axis=0)
    # spectrum = np.expand_dims (l, axis=0)
    K = 1000000
    lcoh_guess = 50e-6
    c = 225563910  # speed of light (m/s)
    x = np.zeros((1, K))
    x[0, :] = np.linspace(0, 4 * lcoh_guess, K)
    t = x / c
    gamma_t2 = np.zeros((1, K))

    delta_f = np.zeros_like(l)
    for i in range(l.shape[-1] - 1):
        delta_f[i] = c / (l[i + 1] - l[i])

    # power spectral density
    G_f = (l ** 2 / c) * spectrum
    # normlaized PSD
    G_f /= np.sum(G_f)

    gamma_t2[0, :] = 2 * np.real(
        np.sum(np.exp(1j * 2 * pi * np.outer((c / l) * G_f * delta_f, t[0, :])), axis=-2))
    gamma_t2[0, :] /= np.amax(gamma_t2[0, :])

    plt.figure()
    plt.plot(x[0, :] * 1e6, gamma_t2[0, :])
    plt.show()
    idx = find_nearest(gamma_t2[0, :], x[0, :], 1 / np.exp(1))
    coherence_lenght = x[0, idx]
    return coherence_lenght


def unwrap1D(phase):
    unwrapped = np.zeros_like(phase)
    unwrapped[0] = phase[0]

    phase *= -1
    carrier = 0
    center = len(phase) // 2
    for i in range(len(phase) - 1):
        cond = phase[i + 1] - phase[i]

        if np.abs(cond) >= np.pi:
            carrier += -2 * np.sign(cond) * np.pi
        unwrapped[i + 1] = phase[i + 1] + carrier

    # for i in range(len(phase)//2-1):
    #     cond = phase[center + i + 1] - phase[center + i]
    #
    #     if np.abs(cond) >= np.pi:
    #         carrier += -2 * np.sign(cond) * np.pi
    #     unwrapped[center + i + 1] = phase[center + i + 1] + carrier

    return unwrapped

def rotate_around_z(Xin, Yin, Ein, phi, linx_in=0, liny_in=0):
    # Copy field
    Ex = Ein

    # original sampling points and field
    old_ny, old_nx = np.shape(Ex)

    # zero padding
    Ex = np.pad(Ex, ((int(old_ny/2), int(old_nx/2)), (int(old_ny/2), int(old_nx/2))), 'constant', constant_values=(0, 0))

    # new sampling points
    ny, nx = np.shape(Ex)

    # extended spatial coordinates and spatial frequencies
    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]
    x1 = np.fft.fftshift(np.fft.fftfreq(nx, 1)) * nx * dx1
    y1 = np.fft.fftshift(np.fft.fftfreq(ny, 1)) * ny * dy1
    X, Y = np.meshgrid(x1, y1)

    # extended spatial frequencies
    dx1 = X[0, 1] - X[0, 0]
    dy1 = Y[1, 0] - Y[0, 0]
    sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    Sx, Sy = np.meshgrid(sx1, sy1)

    # calculation of the shearing parameters
    Shx1 = Y * np.tan(phi / 2)
    Shy1 = X * np.sin(phi)

    # rotation by three shearing transforms
    Gx = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ex)))
    Exm = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gx * np.exp(2 * np.pi * 1j * Shx1 * Sx))))
    
    Gxm = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Exm)))
    Exm = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gxm * np.exp(-2 * np.pi * 1j * Shy1 * Sy))))
    
    Gxm = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Exm)))
    Ex = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gxm * np.exp(2 * np.pi * 1j * Shx1 * Sx))))

    # add analytical linear phase change
    linx_out = linx_in
    liny_out = liny_in

    liny_out = liny_out - linx_out*np.tan(phi/2)
    linx_out = linx_out + liny_out*np.sin(phi)
    liny_out = liny_out - linx_out*np.tan(phi/2)

    # undo zeropadding
    Ex = Ex[int(old_ny/2):int(old_ny/2)+old_ny, int(old_nx/2):int(old_nx/2)+old_nx]
    Xout = Xin
    Yout = Yin
    Eout = Ex
    return Eout, Xout, Yout, linx_out, liny_out

def rotate_around_x(Xin, Yin, Ein, waveLen, phi, Xout, Yout, linx_in=0, liny_in=0):
    # Copy field
    Ex = Ein

    # original sampling points and field
    old_ny, old_nx = Ex.shape

    # zero padding
    Ex = np.pad(Ex, ((int(old_ny/2), int(old_nx/2)), (int(old_ny/2), int(old_nx/2))), 'constant', constant_values=(0, 0))

    # new sampling points
    ny, nx = Ex.shape

    # extended spatial coordinates and spatial frequencies
    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]
    x1 = np.fft.fftshift(np.fft.fftfreq(nx, 1)) * nx * dx1
    y1 = np.fft.fftshift(np.fft.fftfreq(ny, 1)) * ny * dy1
    X, Y = np.meshgrid(x1, y1)
    Z = 0 * X

    # extended spatial frequencies
    dx1 = Xin[0, 1] - Xin[0, 0]
    dy1 = Yin[1, 0] - Yin[0, 0]
    sx1 = np.fft.fftshift(np.fft.fftfreq(nx, dx1))
    sy1 = np.fft.fftshift(np.fft.fftfreq(ny, dy1))
    Sx, Sy = np.meshgrid(sx1, sy1)

    # calculation of the spectrum
    Gxm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
    Kx = Sx * 2 * pi
    Ky = Sy * 2 * pi

    # coupled coordinates z' = z(y)
    z2y = Yout[:, 1] * np.tan(phi) * np.cos(phi)

    # scaled coordinates in Y'
    Ys = Yout * np.cos(phi)

    # output sampling points
    my, mx = np.shape(Xout)

    # analytical phase
    Sx = Sx + linx_in
    Sy = Sy + liny_in

    # parameters for the chirp-z
    dx2 = np.abs(Xout[0, 1] - Xout[0, 0])
    rxb = (dx1 / dx2) * (nx / mx)
    if rxb < 1:
        raise Exception('Zoom out is not possible. Please change the output sampling grid.')

    sx = np.fft.fftshift(np.fft.fftfreq(nx,dx1))
    dkxb = (sx[1] - sx[0])*2*pi
    dxb = 2*pi/(dkxb*nx)
    kxminb = -np.ceil((mx-1)/2)*dxb/rxb
    kxmaxb = np.floor((mx-1)/2)*dxb/rxb
    kxMaxb = mx*dxb
    Ax = np.exp(-1j*pi*(2*kxminb/kxMaxb + 0*dxb/kxMaxb))
    Wx = np.exp(1j*2*pi*(kxmaxb + dxb/rxb - kxminb)/((kxMaxb)*(mx)))

    # calculation for Ex
    Gx2 = np.zeros((my, nx), dtype=complex)
    Mx = Gxm*np.exp(2*np.pi*1j*(Ys[0,0]*Sy + z2y[0]*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
    dMx = np.exp(2*np.pi*1j*((Ys[1,0] - Ys[0,0])*Sy + (z2y[1] - z2y[0])*np.sqrt(1/(waveLen)**2 - Sx**2 - Sy**2)))
    Gx2[0,:] = np.sum(Mx, axis=0)/ny
    for jy in range(my-1):
        Mx = Mx*dMx
        Gx2[jy+1,:] = np.sum(Mx, axis=0)/ny 

    dim2 = 2
    Ex =  chirpz2Daxis(Gx2, Ax, Wx, mx, dim2)/nx

    # analytical linear phase,
    liny_out = liny_in - np.cos(phi) * np.tan(phi)
    linx_out = linx_in

    # substract the linear phase
    Eout = Ex * np.exp(-1j * 2 * pi * Ys * np.tan(phi) / waveLen)

    return Eout, Xout, Yout, linx_out, liny_out


def chirpz2Daxis(U1, A, W, M, dim=1):
    """
    Returns the chirp-z transform of an array along the given axis for specified parameters.
    """

    if dim == 1:
        N = U1.shape[0]
        My = M
        Mx = U1.shape[1]
        Lx = 2 ** int(np.ceil(np.log2(N + M - 1)))
        Ly = 2 ** int(np.ceil(np.log2(N + M - 1)))
        L = Ly

    elif dim == 2:
        N = U1.shape[1]
        Mx = M
        My = U1.shape[0]
        Lx = 2 ** int(np.ceil(np.log2(N + M - 1)))
        Ly = N
        L = Lx

    nn = np.linspace(0, N - 1, N)
    mn = np.linspace(0, M - 1, M)
    ln = np.linspace(L - N + 1, L - 1, N - 1)
    be = np.linspace(M, L - N, L - N - M + 1)

    W1 = W ** (((nn) ** 2) / 2)
    W2 = W ** (-((mn) ** 2) / 2)
    W3 = W ** (-((L - ln) ** 2) / 2)
    W4 = 0 * be

    A1 = A ** (-nn)
    V = np.fft.fft(np.concatenate((W2, W4, W3), axis=0), axis=0)
    Am = A1 * W1
    Vm = V

    # select correct axis
    if dim == 1:
        Am = np.transpose(Am)
        Vm = np.transpose(Vm)
    else:
        Am = Am
        Vm = Vm

    # linear convolution
    Y = np.fft.fft(np.pad(U1 * Am, ((0, Ly - U1.shape[0]), (0, Lx - U1.shape[1]))), axis=dim-1)
    U2g = np.fft.ifft(Y * Vm, axis=dim-1)
    
    #phase factor for centering
    fak = (W2 ** -1) * ((A ** -1) * (W ** (np.linspace(0, M - 1, M))) ** (-1 * (int(np.floor(N / 2)))))
    
    # output field
    U2g_extr = U2g[:My, :Mx]
    U2 = U2g_extr * fak

    return U2

def wavelength_to_rgb(wavelength, gamma=0.8, opacity=200):

    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 800:
        attenuation = 0.3 + 0.7 * (800 - wavelength) / (800 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 1.0
        G = 1.0
        B = 1.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B), opacity)


def spiral_blade_mask(wavelength=13.5e-9, f=0.6e-3, N=256, dx=10e-9, n_blades=3, blades_diameter=8e-6, angle=None):
    """
    :param wavelength: target wavelength
    :param f: focus distance, the smaller --> more twisting of the blades around the center
    :param N: #pixels along 1-direction
    :param dx: pixel space
    :param n_blades: # of blades to generate
    :param blades_diameter: extension of the blades
    :return: binary array NxN where 0 represents the blade structures
    """
    if angle is not None:
        stretching_factor = 1 / np.cos(np.deg2rad(angle))
    else:
        stretching_factor = 1

    x = np.arange(-N / 2, N / 2) * dx
    y = np.copy(x)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid /= stretching_factor
    # y_grid *= stretching_factor
    r = (x_grid ** 2 + y_grid ** 2) ** (1 / 2)
    # r = (x_grid ** 2 + (y_grid*stretching_factor) ** 2) ** (1 / 2)
    # r = ((x_grid/stretching_factor) ** 2 + (y_grid) ** 2) ** (1 / 2)


    phi = np.arctan2(y_grid, x_grid)
    # phi = np.arctan2(y_grid*stretching_factor, x_grid)
    # phi = np.arctan2(y_grid, x_grid/stretching_factor)

    # n_blades = n_blades % 5 + 2  #  minimum amount of blades = 2, increases to 6 anc cycles back
    data = np.exp(-1j * np.pi * r ** 2 / f / wavelength) * np.exp(1j * n_blades * phi)

    binary = np.real(data) < 0
    circ = x_grid ** 2 + y_grid ** 2 < (blades_diameter / 2) ** 2
    # circ = (x_grid/np.sqrt(2)) ** 2 + y_grid ** 2 < (blades_diameter / 2) ** 2

    binary = circ * binary
    binary = (~binary).astype(int)

    return binary


def remove_phase_ramp(myObject):
    # find center of mass
    ftobj = fft2c(myObject) * np.conj(fft2c(myObject))
    ftobj = np.real(ftobj)
    cy, cx = ndi.center_of_mass(ftobj)
    # re_center using fft or
    object_1_centered1 = ifft2c(re_center_ptychogram(fft2c(myObject), center_coord=np.array([cy, cx])))

    # using phase ramp multiplication
    No = myObject.shape[-1]
    xp = np.linspace(-No // 2, No // 2, No)
    Xp, Yp = np.meshgrid(xp, xp)
    xcoord = (Xp - np.amin(Xp))
    ycoord = (Yp - np.amin(Yp))
    thetax = -(cx - No // 2)
    thetay = -(cy - No // 2)
    phase_ramp = np.exp(1.j * (2 * np.pi / No) * xcoord * thetax) * np.exp(1.j * (2 * np.pi / No) * ycoord * thetay)
    object_1_centered2 = myObject * phase_ramp
    return object_1_centered1, object_1_centered2


def crop(data, center_coordinate):
    if center_coordinate[0] < data.shape[0] / 2:
        data = data[:int(round(2 * center_coordinate[0])), :]
    else:
        data = data[int(round(2 * center_coordinate[0] - data.shape[0])):, :]
    if center_coordinate[1] < data.shape[1] / 2:
        data = data[:, :int(round(2 * center_coordinate[1]))]
    else:
        data = data[:, int(round(2 * center_coordinate[1] - data.shape[1])):]
    return data

def re_center_ptychogram(data, center_coord):
    """
    re-centers ptychogram to a given center_coord
    """
    center_coord = np.around(center_coord, decimals=0)#.astype(np.int32)
    shape = data.shape
    centered = np.zeros_like(data)

    if center_coord[0] <= shape[0] / 2:
        ylen = 2 * center_coord[0]
        if center_coord[1] <= shape[1] / 2:
            xlen = 2 * center_coord[1]
        else:
            xlen = (shape[1] - center_coord[1]) * 2
    else:
        ylen = (shape[0] - center_coord[0]) * 2
        if center_coord[1] <= shape[1] / 2:
            xlen = 2 * center_coord[1]
        else:
            xlen = (shape[1] - center_coord[1]) * 2

    cropped_data = crop(data, center_coord)
    ymin = int((shape[0] - ylen) / 2)
    ymax = int((shape[0] + ylen) / 2)
    xmin = int((shape[1] - xlen) / 2)
    xmax = int((shape[1] + xlen) / 2)
    centered[ymin: ymax, xmin: xmax] = cropped_data
    return centered

def cropCenter(ptychogram, size, shift_x=None, shift_y=None):
    '''
    The parameter size corresponds to the finale size of the diffraction patterns
    '''
    if not isinstance(size, int):
        raise TypeError('Crop value is not valid. Int expected')

    x = ptychogram.shape[-1]
    y = ptychogram.shape[-2]
    startx = x // 2 - (size // 2)
    starty = y // 2 - (size // 2)

    startx += 1 if startx > 0 else 0
    starty += 1 if starty > 0 else 0
    if shift_x is not None:
        startx += shift_x
    if shift_y is not None:
        starty += shift_y

    ptychogram = ptychogram[..., starty: starty + size, startx: startx + size]

    return ptychogram

def phase_ramp(slope_x, slope_y, offset, shape):
    """
    Adds a phase rampt to the object.
    :param slope_x: Slope along x direction
    :param slope_y: Slope along y direction
    :param offset: constant phase offset.
    """
    y = np.linspace(-1, 1, shape[0]) * shape[0] * np.pi
    x = np.linspace(-1, 1, shape[1]) * shape[1] * np.pi

    x_grid, y_grid = np.meshgrid(x, y)
    ramp = x_grid * slope_x/shape[1] + y_grid *slope_y/shape[0]

    return ramp + offset

def ringfunc(r, y_shape, x_shape):
    """
    Returns a ring with the radius r.

    :param r: Radius of the ring
    :param y_shape: Shape in y direction of the windw.
    :param x_shape: Shape inx direction of the window
    :return:
    """
    x = np.arange(0, x_shape, 1) - x_shape / 2 + 0.5
    y = np.arange(0, y_shape, 1) - y_shape / 2 + 0.5
    y_grid, x_grid = np.meshgrid(y, x)

    circ_outer = y_grid**2 + x_grid**2 > r**2
    circ_inner = y_grid**2 + x_grid**2 <= (r - 1)**2
    ring = circ_outer + circ_inner
    ring = (ring - 1.) * -1.
    return ring

def one_bit_criterion(N):
    return (0.5 + 2.41/np.sqrt(N)) / (1.5 + 1.41/np.sqrt(N))

def half_bit_criterion(N):
    return (0.2071 + 1.91 / np.sqrt(N)) / (1.2071 + 0.9102 / np.sqrt(N))


def error(reconstruction, simulation):
    return np.sum(np.abs(reconstruction - simulation) ** 2) / np.sum(np.abs(simulation) ** 2)

def FRC(image_1, image_2, filter=True, global_phase_pos=None, filter_radius=100, mask_phase=False, show_difference=False,
        pramp_1=None, pramp_2=None, norm_pos=None):
    """
    Image_1: reconstructed image,
    image_2: Reference
    return frc, one_bit_crit, half_bit_crit, error(image_1, image_2)
    """
    shift_order = 5
    # normalize images to the same avg value
    image_2 /= np.abs(cropCenter(image_2, 2 * filter_radius)).mean()
    image_1 /= np.abs(cropCenter(image_1, 2 * filter_radius)).mean()

    #check error metric before shift
    error_bs = error(image_1, image_2)

    # First check if both images have the same center:
    shift_distance = register_translation(np.abs(image_1), np.abs(image_2), upsample_factor=100)[0]
    print("Shift distance: " + str(shift_distance))
    # shift_distance += np.array([0, .3])
    temp = shift(np.real(image_2), shift_distance, order=shift_order) + 1j * shift(np.imag(image_2), shift_distance, order=shift_order)

    #check error after shift
    error_as = error(image_1, temp)

    if error_as < error_bs:
        image_2 = temp
    image_2 = temp
    if filter:
        image_1 = cropCenter(image_1, 2 * filter_radius)
        image_2 = cropCenter(image_2, 2 * filter_radius)

    print(global_phase_pos)

    if global_phase_pos is not None:
        print("Adjusting global phase")
        # Substracts a global phase offset from a given position for both images
        image_1 *= np.exp(-1j * np.angle(image_1[global_phase_pos]))
        image_2 *= np.exp(-1j * np.angle(image_2[global_phase_pos]))

    if norm_pos is not None:
        image_1 /= np.abs(image_2[norm_pos])
        image_2 /= np.abs(image_2[norm_pos])

    # plt.figure("Original image")
    # plt.imshow(np.abs(image_1))
    #
    # plt.figure("Original image phase")
    # plt.imshow(np.angle(image_1))
    #
    # plt.figure("Image compare")
    # plt.imshow(np.abs(image_2))
    #
    # plt.figure("Image compare phase")
    # plt.imshow(np.angle(image_2))

    if pramp_1 is not None:
        p_ramp = phase_ramp(pramp_1[0], pramp_1[1], 0, image_1.shape)
        image_1 *= np.exp(1j * p_ramp)

    if pramp_2 is not None:
        p_ramp = phase_ramp(pramp_2[0], pramp_2[1], 0, image_2.shape)
        image_2 *= np.exp(1j * p_ramp)

    print("--------------")
    print("FRC")

    y_shape = image_1.shape[0]
    x_shape = image_1.shape[1]
    R = image_1.shape[0]/2
    window_func = np.hanning(y_shape).reshape(1, -1) * np.hanning(x_shape).reshape(-1, 1)

    fft_image1 = fftshift(fft2(fftshift(image_1 * window_func)))
    fft_image2 = fftshift(fft2(fftshift(image_2 * window_func)))
    fft_image1 /= np.max(np.abs(fft_image1))
    fft_image2 /= np.max(np.abs(fft_image2))
    conj_image1 = np.conj(fft_image1)

    frc = np.zeros(int(np.min([x_shape, y_shape])/2), dtype=complex)
    one_bit_crit = np.zeros_like(frc)
    half_bit_crit = np.zeros_like(frc)

    # plt.figure()
    # plt.subplot(131)
    # plt.title("Fourier recon")
    # plt.imshow(np.log10(np.abs(fft_image1)))
    #
    # plt.subplot(132)
    # plt.title("Fourier reference")
    # plt.imshow(np.log10(np.abs(fft_image2)))
    #
    # plt.subplot(133)
    # plt.title("Differences")
    # plt.imshow(np.abs(fft_image2 - fft_image1))

    if show_difference:
        plt.figure()
        plt.subplot(121)
        plt.title("Phase difference abs. value")
        plt.imshow(np.abs(np.angle(image_1) - np.angle(image_2)), interpolation="none", vmin=0, vmax=1)
        plt.subplot(122)
        plt.title("Ampl. difference")
        plt.imshow(np.abs(image_1) - np.abs(image_2), interpolation="none")

    max = np.max(np.abs(image_1)) / 3.

    # if mask_phase:
    #     plt.figure()
    #     plt.imshow(np.angle(np.where(np.abs(image_1) > max, image_1, 0)))
    # else:
    #     plt.figure()
    #     plt.imshow(np.angle(image_1))

    for r in range(int(R)):
        r += 1
        # print('x shape')
        # print(x_shape)
        ring = ringfunc(r, x_shape, y_shape)
        one_bit_crit[r-1] = one_bit_criterion(np.sum(ring))
        half_bit_crit[r-1] = half_bit_criterion(np.sum(ring))
        # check complex value/real value
        frc[r-1] = np.sum(ring * conj_image1 * fft_image2) / np.sqrt(np.sum(ring * np.abs(fft_image1)**2) * np.sum(ring * np.abs(fft_image2)**2))

    return frc, one_bit_crit, half_bit_crit, error(image_1, image_2)


class MyFRC:
    """
    Example to use:
    myFRC = MyFRC(object_1, object_2, dx)
    myFRC.show_raw_data()
    myFRC.normalize_amplitude()
    myFRC.remove_phase_ramp()
    myFRC.show_comparison_after_phase_ramp_removal(id=0)
    myFRC.show_comparison_after_phase_ramp_removal(id=1)

    #based on plotted results, choose the best result for each object
    myFRC.choose_phase_ramp_result(id=0, result=0)
    myFRC.choose_phase_ramp_result(id=1, result=0)

    region = slice(0,50), slice(0,50)
    myFRC.remove_global_phase_from_avg_region(region)

    myFRC.align_objects()
    myFRC.show_centered_objects()
    myFRC.clip_filter_objects(filter_radius=180)
    myFRC.show_clipped_objects()
    myFRC.calculateFRC()
    myFRC.plotFRC()
    myFRC.get_spatial_resolution()
    """
    def __init__(self, object1, object2, dx):
        self.object1 = object1
        self.object2 = object2
        self.dx = dx
        self._match_shape()

    def _match_shape(self):
        s1 = self.object1.shape
        s2 = self.object2.shape
        min_shape = min(s1,s2)
        if s1 != min_shape:
            # crop obj1 to obj2's shape
            self.object1 = cropCenter(self.object1,size=min_shape[0])  
        if s2 != min_shape:
            #crop obj2 to obj1's shape
            self.object2 = cropCenter(self.object2, size=min_shape[0])
    
    def show_raw_data(self):
        # plot raw data
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        fig.suptitle('raw files')
        axes[0].imshow(complex2rgb(self.object1))
        axes[0].set_axis_off()
        axes[1].imshow(complex2rgb(self.object2))
        axes[1].set_axis_off()
        fig.canvas.draw()#(block=False)

    def normalize_amplitude(self):
        # normalized amplitude
        # self.object1 /= np.sqrt(np.abs(self.object1)).mean()
        # self.object2 /= np.sqrt(np.abs(self.object2)).mean()
        # self.object1 /= np.mean(np.abs(self.object1))
        # self.object2 /= np.mean(np.abs(self.object2))

        # self.object1 /= np.amax(np.abs(self.object1))
        # self.object2 /= np.amax(np.abs(self.object2))
        # print(np.abs(self.object1).mean())
        # print(np.abs(self.object2).mean())

        N = self.object1.shape[-1] // 2
        W = int(N * 0.4)
        self.object1 /= np.abs(self.object1[N-W//2:N+W//2,N-W//2:N+W//2]).mean()
        self.object2 /= np.abs(self.object2[N-W//2:N+W//2,N-W//2:N+W//2]).mean()

    def normalize_amplitude2(self):
        N = self.object1p.shape[-1] // 2
        W = int(N * 0.4)
        self.object1p /= np.abs(self.object1p[N-W//2:N+W//2,N-W//2:N+W//2]).mean()
        self.object2p /= np.abs(self.object2p[N-W//2:N+W//2,N-W//2:N+W//2]).mean()

    def remove_global_phase_from_point(self, px=0, py=0):
        # same phase
        ref_phase = np.angle(self.object1p[py, px])
        self.object1p *= np.exp(-1j * ref_phase)
        ref_phase = np.angle(self.object2p[py, px])
        self.object2p *= np.exp(-1j * ref_phase)

    def remove_global_phase_from_avg_region(self, region):
        # same phase
        ref_phase = np.angle(self.object1p[region])
        ref_phase = np.mean(ref_phase)
        self.object1p *= np.exp(-1j * ref_phase)
        ref_phase = np.angle(self.object2p[region])
        ref_phase = np.mean(ref_phase)
        self.object2p *= np.exp(-1j * ref_phase)

    def remove_phase_ramp(self):
        # remove phase ramp
        self.obj1_v1, self.obj1_v2 = remove_phase_ramp(self.object1)
        self.obj2_v1, self.obj2_v2 = remove_phase_ramp(self.object2)
        self.results_phase_ramp = [[self.object1, self.obj1_v1, self.obj1_v2],
                                   [self.object2, self.obj2_v1, self.obj2_v2]]

    def show_comparison_after_phase_ramp_removal(self, id=0):
        if id == 0:  # show for obj1    
            before = self.object1
            v1 = self.obj1_v1
            v2 = self.obj1_v2
        if id == 1:  # show for obj1    
            before = self.object2
            v1 = self.obj2_v1
            v2 = self.obj2_v2

        # show comparison raw, centered 1, centered 2
        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        fig.suptitle(f'phase ramp removal object {id+1:.0f}')
        axes[0].set_title('original')
        axes[0].imshow(complex2rgb(before))
        axes[0].set_axis_off()
        axes[1].set_title('V1')
        axes[1].imshow(complex2rgb(v1))
        axes[1].set_axis_off()
        axes[2].set_title('V2')
        axes[2].imshow(complex2rgb(v2))
        axes[2].set_axis_off()
        fig.tight_layout()
        fig.show()

    def choose_phase_ramp_result(self, id=0, result=0):
        """
        :param id: 0 for 0bjec1, 1 for object 2 
        :param result: 0,1,2 [original, v1, v2] see plot_comparison_after_phase_ramp_removal
        :return: None
        """
        if id == 0:
            self.object1p = self.results_phase_ramp[id][result]
        if id == 1:
            self.object2p = self.results_phase_ramp[id][result]

    def align_objects(self):
        N = self.object1p.shape[-1] // 2
        W = int(N * 1)
        region = np.zeros_like(self.object1p)
        region[N-W//2:N+W//2,N-W//2:N+W//2] = 1.0
        object_2 = self.object1p #* region
        object_1 = self.object2p #* region

        # align objects
        # check error metric before shift
        error_bs = error(self.object1p, self.object2p)
        # First check if both images have the same center:
        # check using amplitude
        shift_distance1 = register_translation(np.abs(object_1), np.abs(object_2), upsample_factor=100)[0]
        print("Shift distance amp: " + str(shift_distance1))
        # check using phase
        shift_distance2 = register_translation(np.angle(object_1), np.angle(object_2), upsample_factor=100)[0]
        print("Shift distance phase: " + str(shift_distance2))
        shift_distance = -shift_distance1
        shift_distance2 = -shift_distance2

        # shift_distance += np.array([0, .3])
        temp = shift(np.real(self.object2p), shift_distance, order=5) + 1j * shift(np.imag(self.object2p),
                                                                                    shift_distance,
                                                                                    order=5)
        temp2 = shift(np.real(self.object2p), shift_distance2, order=5) + 1j * shift(np.imag(self.object2p),
                                                                                   shift_distance2,
                                                                                   order=5)

        # check error after shift
        error_as = error(self.object1p, temp)
        error_ph = error(self.object1p, temp2)
        print(f'error before shift: {error_bs}\n'
              f'error after shift (amp): {error_as}\n'
              f'error after shift (phase): {error_ph}')
        if error_as < error_bs or error_ph < error_bs:
            if error_as < error_ph:
                self.object2p = temp
            else:
                self.object2p = temp2

        # self.object2p = temp

    def show_centered_objects(self):
        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        fig.suptitle('aligned objects')
        axes[0].imshow(complex2rgb(self.object1p))
        axes[0].set_axis_off()
        axes[1].imshow(complex2rgb(self.object2p))
        axes[1].set_axis_off()
        axes[2].set_title('abs difference')
        # difference = np.abs(self.object1p)**2 - np.abs(self.object2p)**2 / np.amax(np.abs(self.object1p)**2)
        difference = np.abs(self.object1p - self.object2p) ** 2 / np.amax(np.abs(self.object1p)**2)#- np.abs(self.object2p) ** 2  # / np.abs(self.object2p)**2
        # difference = np.abs(self.object1p)**2 - np.abs(self.object2p)**2 / \
        #              (np.abs(self.object1p)**2 + np.abs(self.object2p)**2)

        im=axes[2].imshow(difference, cmap='twilight')
        axes[2].set_axis_off()
        fig.colorbar(im, ax=axes[2], shrink=0.9)
        fig.tight_layout()
        fig.show()

    def clip_filter_objects(self, filter_radius=None):
        N = self.object1p.shape[-1] // 2
        if filter_radius is None:
            # clip/filter region
            filter_radius = int(N * 0.4)
            self.object_1c = self.object1p
            self.object_2c = self.object2p
        else:
            # self.object_1c = self.object1p*rect_px(self.object1p.shape[-1], filter_radius * 2)
            # self.object_2c = self.object2p*rect_px(self.object1p.shape[-1], filter_radius * 2)
            self.object_1c = cropCenter(self.object1p, 2 * filter_radius)
            self.object_2c = cropCenter(self.object2p, 2 * filter_radius)

    def show_clipped_objects(self):
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        fig.suptitle('clipped objects')
        axes[0].imshow(complex2rgb(self.object_1c))
        axes[0].set_axis_off()
        axes[1].imshow(complex2rgb(self.object_2c))
        axes[1].set_axis_off()
        fig.tight_layout()
        fig.show()
    
    def calculateFRC(self):
        y_shape = self.object_1c.shape[0]
        x_shape = self.object_1c.shape[1]
        R = self.object_1c.shape[0] / 2
        
        window_func = np.hanning(y_shape).reshape(1, -1) * np.hanning(x_shape).reshape(-1, 1)
        
        fft_image1 = fftshift(fft2(fftshift(self.object_1c * window_func)))
        fft_image2 = fftshift(fft2(fftshift(self.object_2c * window_func)))
        fft_image1 /= np.max(np.abs(fft_image1))
        fft_image2 /= np.max(np.abs(fft_image2))
        conj_image1 = np.conj(fft_image1)

        self.frc = np.zeros(int(np.min([x_shape, y_shape]) / 2), dtype=complex)
        self.one_bit_crit = np.zeros_like(self.frc)
        self.half_bit_crit = np.zeros_like(self.frc)
    
        for r in range(int(R)):
            r += 1
            ring = ringfunc(r, x_shape, y_shape)
            self.one_bit_crit[r-1] = one_bit_criterion(np.sum(ring))
            self.half_bit_crit[r-1] = half_bit_criterion(np.sum(ring))
            # check complex value/real value
            self.frc[r-1] = np.sum(ring * conj_image1 * fft_image2) / np.sqrt(np.sum(ring * np.abs(fft_image1)**2) * np.sum(ring * np.abs(fft_image2)**2))

    def plotFRC(self):
        n_ticks = 10
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=150)
        fig.suptitle(f'FRC')
        ax.plot(self.frc, label=f'FRC')
        ax.plot(self.half_bit_crit, '--', label='1/2 bit')
        self.qmax = 1 / (2 * self.dx * 1e6)
        self.Fx = np.round(np.linspace(0, self.qmax, n_ticks), decimals=3)
        plt.xticks(np.linspace(0, len(self.frc), n_ticks), labels=self.Fx)
        ax.set_ylabel('FRC')
        ax.set_xlabel(r'Spatial freq. ($\mu m^{-1}$)')
        ax.grid(alpha=0.5)
        ax.minorticks_on()
        ax.legend()
        fig.tight_layout()
        fig.show()

    def get_FRC_plot_data(self):
        return self.frc, self.half_bit_crit, self.dx

    def get_spatial_resolution(self):
        #half bit criteria
        qm = np.linspace(0, self.qmax, self.frc.shape[-1])
        index = np.argwhere(self.frc < self.half_bit_crit)
        if len(index) > 0:
            index= index[0]
            res = 1 / (2 * qm[index])  # (um)
        else:
            res = 1 / (2 * qm[-1])  # (um)
        print(f'resolution: {res} um')
        return res


def simulate_ccd_image(field, bit_depth=12, peak_photons=10000, quantum_efficiency=0.7,
                       quantum_well=30000, readout_noise=10, dc_level=100):
    """
    Simulate CCD sensor image capture with noise and DC level.

    Parameters:
    field : 2D complex numpy array
        The complex field incident on the sensor
    bit_depth : int
        Bit depth of the camera (default: 12)
    peak_photons : float
        Maximum number of photons in the brightest pixel (default: 10000)
    quantum_efficiency : float
        Quantum efficiency of the sensor (default: 0.7)
    quantum_well : int
        Full well capacity of the pixel (default: 30000)
    readout_noise : float
        Standard deviation of readout noise in electrons (default: 10)
    dc_level : float
        DC level in electrons, representing dark current and fixed pattern noise (default: 100)

    Returns:
    2D numpy array of uint16
        Simulated CCD image
    """

    # Calculate intensity (proportional to photon count)
    intensity = np.abs(field) ** 2

    # Scale intensity to peak_photons
    scale_factor = peak_photons / np.max(intensity)
    photon_count = intensity * scale_factor

    # Apply quantum efficiency
    electron_count = np.random.poisson(photon_count * quantum_efficiency)

    # Add DC level (with Poisson noise)
    electron_count += np.random.poisson(dc_level, electron_count.shape)

    # Apply quantum well limitation
    if quantum_well is not None:
        electron_count = np.minimum(electron_count, quantum_well)

    # Add readout noise
    electron_count = electron_count + np.random.normal(0, readout_noise, electron_count.shape)

    # Convert to ADU (Analog-to-Digital Units)
    max_adu = 2 ** bit_depth - 1
    if quantum_well is not None:
        adu_per_electron = max_adu / quantum_well
    else:
        adu_per_electron = max_adu / np.amax(electron_count)

    adu_count = electron_count * adu_per_electron

    # Quantize and clip
    adu_count = np.clip(np.round(adu_count), 0, max_adu)

    return adu_count.astype(np.uint16)

if __name__ == "__main__":
    pass