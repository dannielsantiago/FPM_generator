import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


def zernike_radial(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial.
    """
    if (n - m) % 2:
        return rho * 0.0
    radial_poly = 0.0
    for k in range((n - m) // 2 + 1):
        radial_poly += rho ** (n - 2 * k) * (-1) ** k * factorial(n - k) / (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
    return radial_poly


def zernike(m, n, rho, theta):
    """
    Calculate the Zernike polynomial.
    """
    if m > 0:
        return zernike_radial(m, n, rho) * np.cos(m * theta)
    elif m < 0:
        return zernike_radial(-m, n, rho) * np.sin(-m * theta)
    else:
        return zernike_radial(m, n, rho)


def zernike_polynomial(m, n, rho, theta):
    """
    Generate a Zernike polynomial over a circular aperture.
    """
    Z = np.zeros_like(rho)
    aperture = rho <= 1
    Z[aperture] = zernike(m, n, rho[aperture], theta[aperture])
    return Z


def combined_zernike(coefficients, npix=256, N=512):
    """
    Generate a combined Zernike polynomial based on a set of coefficients.
    """
    assert N >= npix, "N should be greater than or equal to npix"

    x = np.linspace(-1, 1, npix)
    y = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    Z_combined_small = np.zeros_like(X)
    # Coefficients should be provided as a list of tuples: (m, n, coefficient)
    for (m, n, coeff) in coefficients:
        Z_combined_small += coeff * zernike_polynomial(m, n, rho, theta)

    # Embed the smaller Zernike pattern into a larger array
    Z_combined_large = np.zeros((N, N))
    start_idx = (N - npix) // 2
    Z_combined_large[start_idx:start_idx + npix, start_idx:start_idx + npix] = Z_combined_small

    return Z_combined_large