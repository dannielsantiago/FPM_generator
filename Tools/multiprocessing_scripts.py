import numpy as np
from Tools.propagators import propagate

def propagate_(field, dx, dz, wavelength, method = 'aspw'):
    u1 = propagate(field, dx=dx, dz=dz, method=method, wavelength=wavelength)
    return np.abs(u1).astype(np.float32)

def RS_diffraction_integral(args):
    """
    Compute the Rayleigh-Sommerfeld diffraction integral using a vectorized implementation.

    Parameters:
        U_source (array): Complex amplitude of the wave at the source plane.
        Xs (array): X-coordinates on the source plane.
        Ys (array): Y-coordinates on the source plane.
        x (float): X-coordinate on the observation plane.
        y (float): Y-coordinate on the observation plane.
        z (float): Propagation distance.
        wavelength (float): Wavelength of the wave.

    Returns:
        complex: Complex amplitude of the wave at the observation plane.
    """
    U_source, Xs, Ys, x, y, z, wavelength, dx = args
    k = 2 * np.pi / wavelength
    r = np.sqrt((x - Xs)**2 + (y - Ys)**2 + z**2)
    phase_term = np.exp(1j * k * r) * z / r**2
    second_term = 1/(2 * np.pi) - 1j/wavelength
    U_observation = np.sum(U_source * phase_term * second_term * dx**2)
    return U_observation


def RS_point_source_to_plane(x_source, y_source, X_dest, Y_dest, z, wavelength):
    """
    Compute the Rayleigh-Sommerfeld diffraction from a point source to a 2D plane.

    Parameters:
        x_source, y_source (float): Coordinates of the point source
        X_dest, Y_dest (2D arrays): Meshgrid of x and y coordinates on the destination plane
        z (float): Propagation distance
        wavelength (float): Wavelength of the wave

    Returns:
        2D array: Complex amplitude of the wave at the destination plane
    """
    k = 2 * np.pi / wavelength

    # Calculate the distance from source to each point on the destination plane
    r = np.sqrt((X_dest - x_source) ** 2 + (Y_dest - y_source) ** 2 + z ** 2)

    # # Calculate the RS integral
    phase_term = np.exp(1j * k * r) * z / r ** 2
    second_term = 1 / (2 * np.pi) - 1j / wavelength

    U_dest = phase_term * second_term

    # Calculate the RS integral
    # phase_term = np.exp(1j * k * r)
    # amplitude_term = (1j * k / r - 1 / r ** 2) * (z / r)
    #
    # U_dest = (1 / (2 * np.pi)) * amplitude_term * phase_term


    return U_dest