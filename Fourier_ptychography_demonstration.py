import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Qt5agg")
matplotlib.use("QtAgg")

from Tools.propagators import fft2c, ifft2c
from Tools.misc import complex2rgb, circ_px, CMAP_DIFFRACTION, add_complex_colorwheel, wavelength_to_rgb, simulate_ccd_image
from Tools.zernike_polynomials import *
from matplotlib.patches import Circle
from Tools.multiprocessing_scripts import RS_diffraction_integral, RS_point_source_to_plane

"""
Load image as sample
"""
# Load an image and normalize it
my_object_RGB = plt.imread('imgs/PiotrZakrzewski_5197202.png')  # RGB image
# normalize its amplitude
my_object_amp = np.mean(my_object_RGB, axis=-1)
my_object_amp /= np.amax(my_object_amp)
#
max_apmplitude_decay = 0.4  # 0: transparent, 1:high-contrast
my_object_amp = (1 - max_apmplitude_decay) + max_apmplitude_decay*my_object_amp
# adds phase information
my_object_phase = np.mean(my_object_RGB, axis=-1)
my_object_phase = abs(my_object_phase - np.amax(my_object_phase))
my_object_phase /= np.amax(my_object_phase)
# constructs complex-valued object
phase_offset = -0.20  # used to correct backgroung color in complex-valued plot
my_object = my_object_amp*np.exp(-1j*2*np.pi*(my_object_phase+phase_offset))

"""
Define experimental parameters
"""
# Define my illumination grid (Matriz de LED)
nLEDs = 10
dl = 1e-3  # Led separation distance
z0 = 15e-2  # Distance between LEDs and sample
wavelength = 630e-9  #LED wavelength illumination

L_led = nLEDs * dl  # lateral extension of led matrix
lx = np.linspace(-nLEDs/2, nLEDs/2, nLEDs)*dl
LX, LY = np.meshgrid(lx, lx)  # 2d- grid coordinates
LED_color = wavelength_to_rgb(wavelength*1e9)
LED_color_normalized = [x/255 for x in LED_color]  # Converted to 0-1 range, with alpha=1.0

# Detection parameters
NA = 0.2  # Numerical aperture
No = my_object.shape[-1]  # Asumming square object
# create lens pupil
Np = int(NA * No)
lens_pupil = circ_px(Np, Np)
# optional, add aberrations to the lens pupil
add_aberrations = False
if add_aberrations:
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
    zernike_poly_combined = combined_zernike(coefficients, npix=Np, N=Np)
    phase_aberration = zernike_poly_combined
    # Complex transmission function
    lens_pupil = lens_pupil*np.exp(1j * phase_aberration)


#sample coordinates
dx = wavelength/(2*NA)  # maximum resolution, this will serve as our pixel-size
L = No * dx  # sample's lateral size in meters
k0 = 2 * np.pi / wavelength
# real space coordinates
x = np.arange(-No / 2, No / 2) * dx
X, Y = np.meshgrid(x, x)
# fourier space coordinates
f = np.arange(-No / 2, No / 2) / L
FX, FY = np.meshgrid(f, f)

"""
Displays LED matrix and sample
"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3.5))
ax1 = ax[0]
ax2 = ax[1]
ax1.set_title('LED matrix')
color_list = [(0.5,0.5,0.5,1)]*nLEDs**2
ax1.scatter(LX*1e2, LY*1e2, marker='s', c=color_list)
ax1.set_xlabel('(cm)')
ax1.set_ylabel('(cm)')
ax1.set_aspect('equal')
ax1.minorticks_on()
ax1.grid(True, alpha=0.5)

ax2.set_title('Complex-valued object')
ax2.pcolormesh(X * 1e3, Y * 1e3, np.ones(shape=(No, No)), color=complex2rgb(my_object).reshape(-1, 3) / 255)
add_complex_colorwheel(fig, ax2, loc=4, pad=0.02)
ax2.set_aspect('equal')
ax2.set_xlabel('(mm)')
ax2.set_ylabel('(mm)')

fig.tight_layout()
# fig.show()
plt.show()

"""
Displays LED illumination effect on sample
"""
single_LED_id = 2
ids = np.arange(0, 100, 2, dtype=int)
# for single_LED_id in ids:
#     LED_coord_x = LX.flatten()[single_LED_id]
#     LED_coord_y = LY.flatten()[single_LED_id]
#
#     # evaluate RS integral to compute illumination wavefront that will interact with the sample
#     illu_wavefront = RS_point_source_to_plane(LED_coord_x, LED_coord_y, X, Y, z0, wavelength,)
#     # Calculate the total energy
#     total_energy = np.sum(np.square(np.abs(illu_wavefront)))
#     # Normalize the wavefront
#     illu_wavefront = illu_wavefront / np.sqrt(total_energy)
#     # computes FFT of object*illumination
#     my_object_illuminated = my_object * illu_wavefront
#     my_sample_FT = fft2c(my_object_illuminated)
#
#     #create slices to select clipped area by the NA in the fourier space
#     p1 = slice(int(No / 2 - Np / 2), int(No / 2 + Np / 2))
#     p2 = slice(int(No / 2 - Np / 2), int(No / 2 + Np / 2))
#
#     # clip Fourier space and apply with the lens pupil that can
#     # include aberrations
#     my_sample_FT_clipped = my_sample_FT[p1,p2] * lens_pupil
#
#     # FFT of the clipped array and computes the intensity of the field
#     # i.e. what the camera sees:
#     my_image = fft2c(my_sample_FT_clipped)
#     my_detected_image = np.square(np.abs(my_image))
#
#     # additionally here one can define the noise parameters, photon-count, and bith-depth for discretization
#     # of the measured image
#     my_detected_image_with_noise = simulate_ccd_image(my_detected_image,
#                                            bit_depth=12,
#                                            peak_photons=10e6,
#                                            quantum_efficiency=0.7,
#                                            quantum_well=None,
#                                            readout_noise=10,
#                                            dc_level=100)
#
#     """
#     Displays decteted image with and witout noise
#     """
#     # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3.5))
#     # ax1 = ax[0]
#     # ax2 = ax[1]
#     # ax1.set_title('my image')
#     # ax1.imshow(my_detected_image, cmap='gray')
#     # ax1.set_axis_off()
#     # ax1.set_aspect('equal')
#     #
#     # ax2.set_title('my image with noise')
#     # ax2.imshow(my_detected_image_with_noise, cmap='gray')
#     # ax2.set_aspect('equal')
#     # ax2.set_axis_off()
#     #
#     # fig.tight_layout()
#     # fig.show()
#
#
#     # create image coordinates, since the clipped array has a smaller size in pixels as the initial object
#     Ni = Np
#     dxi = L/Ni
#     x = np.arange(-Ni / 2, Ni / 2) * dxi
#     Xi, Yi = np.meshgrid(x, x)
#     f = np.arange(-Ni / 2, Ni / 2) / L
#     Fxi, Fyi = np.meshgrid(f,f)
#
#
#     def plot_my_data():
#         width_pixels = 512
#         height_pixels = 512
#         dpi = 100
#
#         figsize = (width_pixels / dpi, height_pixels / dpi)
#
#         fig, axis = plt.subplots(nrows=2, ncols=2, figsize=figsize)
#         ax = axis.flatten()
#         ax1 = ax[0]
#         ax2 = ax[1]
#         ax3 = ax[2]
#         ax4 = ax[3]
#         ax1.set_title('LED matrix')
#         color_list = [(0.5, 0.5, 0.5, 1)] * nLEDs ** 2
#         ax1.scatter(LX * 1e2, LY * 1e2, marker='s', c=color_list)
#         ax1.scatter(LED_coord_x * 1e2, LED_coord_y * 1e2, marker='s', c=LED_color_normalized)
#         ax1.set_xlabel('(cm)')
#         ax1.set_ylabel('(cm)')
#         ax1.set_aspect('equal')
#         ax1.minorticks_on()
#         ax1.grid(True, alpha=0.5)
#         ax2.set_title('Object illuminated')
#         ax2.pcolormesh(X * 1e3, Y * 1e3, np.ones(shape=(No, No)),
#                        color=complex2rgb(my_object_illuminated).reshape(-1, 3) / 255)
#         add_complex_colorwheel(fig, ax2, loc=4, pad=0.02)
#         ax2.set_aspect('equal')
#         ax2.set_xlabel('(mm)')
#         ax2.set_ylabel('(mm)')
#         ax3.set_title('Fourier space')
#         ax3.pcolormesh(FX / k0, FY / k0, np.log(abs(my_sample_FT) + 0.5), cmap=CMAP_DIFFRACTION)
#         # Add a dashed circle to represnet the cutted region by the NA of the lens
#         circle_radius = (Np / L) / 2 / k0
#         circle = Circle((0, 0), circle_radius, fill=False, linestyle='--', edgecolor='white', linewidth=2, label='NA')
#         ax3.add_patch(circle)
#         ax3.legend()
#         ax3.set_aspect('equal')
#         ax3.set_xlabel('(k0)')
#         ax3.set_ylabel('(k0)')
#         ax4.set_title('Recorded image')
#         ax4.pcolormesh(Xi * 1e3, Yi * 1e3, my_detected_image_with_noise, cmap='gray')
#         ax4.set_aspect('equal')
#         ax4.set_xlabel('(mm)')
#         ax4.set_ylabel('(mm)')
#         fig.tight_layout()
#         # plt.savefig(f'myplot_{single_LED_id}.png', dpi=dpi, bbox_inches='tight')
#
#         # fig.show()
#         plt.show()
#         # plt.close(fig)
#
#     plot_my_data()


