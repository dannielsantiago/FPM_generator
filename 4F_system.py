import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5agg")
from Tools.propagators import fft2c, ifft2c
from Tools.misc import complex2rgb, circ_px, CMAP_DIFFRACTION
from Tools.zernike_polynomials import *

"""
Ideal 4f system as two consecutive FFT
"""
# Load an image and normalize it
my_object = plt.imread('imgs/James_Clerk_Maxwell.png')  # RGB image
my_object = np.mean(my_object, axis=-1)
my_object /= np.amax(my_object)

# first FFT, complex-value data
my_Fourier_space = fft2c(my_object)
# image plane
my_image = fft2c(my_Fourier_space)


# Plot results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,2.5))
fig.suptitle('Ideal 4f-system')

ax[0].set_title('my object')
ax[0].imshow(my_object, cmap='gray')
ax[1].set_title('Fourier space')
ax[1].imshow(complex2rgb(my_Fourier_space, scalling=0.05))
ax[2].set_title('image plane')
ax[2].imshow(abs(my_image), cmap='gray')
for axis in ax:
    axis.set_axis_off()
fig.tight_layout()
fig.show()

"""
4f-system considering a specific Numerical Aperture from a lens (NA)
"""
NA = 0.9 # from 0 to 1
N = my_object.shape[-1]
D = N*NA
my_lens = circ_px(N, D)
my_Fourier_space2 = my_Fourier_space*my_lens
my_image2 = fft2c(my_Fourier_space2)

# Plot results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,2.5))
fig.suptitle(f'4f-system for a NA={NA:.1f}')
ax[0].set_title('my object')
ax[0].imshow(my_object, cmap='gray')
ax[1].set_title('Fourier space')
ax[1].imshow(complex2rgb(my_Fourier_space2, scalling=0.05))
# ax[1].pcolormesh(np.angle(my_Fourier_space2), cmap=CMAP_DIFFRACTION)

ax[2].set_title('image plane')
ax[2].imshow(abs(my_image2), cmap='gray')
for axis in ax:
    axis.set_axis_off()
fig.tight_layout()
fig.show()

"""
physical 4f-system, considering diffraction-limited resolution and aberrations
"""

NA = 0.9  # from 0 to 1
wavelength = 550e-9  # Green light
dx = wavelength/(2*NA)  # resolution, this will serve as our pixel-size
N = my_object.shape[-1]
M = my_object.shape[-2]
x = np.linspace(-N//2, N//2, N)*dx
y = np.linspace(-M//2, M//2, M)*dx
# create 2D-grid coordinates for the object
X, Y = np.meshgrid(x,y)
# creates 2D-grid coordinates for the Fourier space
fx = np.linspace(-N//2, N//2, N)/(dx*N)
fy = np.linspace(-M//2, M//2, M)/(dx*M)
Fx, Fy = np.meshgrid(fx, fy)


D = N*NA
my_lens = circ_px(N, D)
# Define Zernike coefficients (m, n, coefficient)
coefficients = [
    (0, 0, 0),  # Piston
    (1, 1, 0),  # Tilt X
    (1, -1, 0),  # Tilt Y
    (2, 0, 0),  # Defocus
    (2, 2, 50),  # Astigmatism 45°
    (2, -2, 0),  # Astigmatism 0°
]
# Generate the combined Zernike polynomial
zernike_poly_combined = combined_zernike(coefficients, npix=int(D), N=N)
phase_aberration = zernike_poly_combined

# Complex transmission function
my_lens_with_aberrations = my_lens*np.exp(1j * phase_aberration)

# Display the phase profile of the aberrated lens
plt.figure(figsize=(3, 3))
plt.pcolormesh(Fx, Fy, np.angle(my_lens_with_aberrations), cmap='hsv', vmin=-np.pi, vmax=np.pi)
plt.colorbar(label='Phase (radians)')
plt.title('Phase Profile with Aberrations')
plt.show()

my_Fourier_space2 = my_Fourier_space*my_lens_with_aberrations
my_image2 = fft2c(my_Fourier_space2)

# Plot results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,2.5))
fig.suptitle(f'4f-system for a NA={NA:.1f}')
ax[0].set_title('my object')
ax[0].pcolormesh(X,Y, my_object, cmap='gray')
ax[0].set_xlabel('(m)')
ax[0].set_ylabel('(m)')

ax[1].set_title('Fourier space')
ax[1].pcolormesh(Fx, Fy, np.log10(np.abs(my_Fourier_space2)**0.25+1), cmap=CMAP_DIFFRACTION)
# ax[1].pcolormesh(Fx, Fy, np.angle(my_Fourier_space2), cmap=CMAP_DIFFRACTION)
ax[1].set_xlabel('(1/m)')
ax[1].set_ylabel('(1/m)')

ax[2].set_title('image plane')
ax[2].pcolormesh(X,Y,abs(my_image2), cmap='gray')
ax[2].set_xlabel('(m)')
ax[2].set_ylabel('(m)')

for axis in ax:
    axis.set_aspect('equal')

fig.tight_layout()
fig.show()




