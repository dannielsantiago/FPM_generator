import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Qt5agg")
from PtyLab.utils.visualisation import show3Dslider
from Tools.propagators import fft2c, ifft2c
from Tools.misc import complex2rgb, circ_px, CMAP_DIFFRACTION, add_complex_colorwheel, wavelength_to_rgb, simulate_ccd_image
from Tools.zernike_polynomials import *
from matplotlib.patches import Circle
from Tools.multiprocessing_scripts import RS_diffraction_integral, RS_point_source_to_plane
import imageio.v2 as imageio
import datetime
import os
import h5py

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
nLEDs_x = 10
nLEDs_y = 10
dl = 1e-3  # Led separation distance
z0 = 15e-2  # Distance between LEDs and sample
wavelength = 630e-9  #LED wavelength illumination

L_led_x = nLEDs_x * dl  # lateral extension of led matrix
lx = np.linspace(-nLEDs_x/2, nLEDs_x/2, nLEDs_x)*dl
ly = np.linspace(-nLEDs_y/2, nLEDs_y/2, nLEDs_y)*dl
LX, LY = np.meshgrid(lx, ly)  # 2d- grid coordinates
LED_color = wavelength_to_rgb(wavelength*1e9)
LED_color_normalized = [x/255 for x in LED_color]  # Converted to 0-1 range, with alpha=1.0

# Detection parameters
NA = 0.2  # Numerical aperture
No = my_object.shape[-1]  # Asumming square object
# create lens pupil
Np_inner = int(NA * No)
# List of threshold values. This ensures that the final images are power of 2
thresholds = [256, 512, 1024, 2048]
# Calculate Np based on the value of No and NA
Np = int(NA * No)
# Find the next threshold value greater than or equal to Np
for threshold in thresholds:
    if Np_inner < threshold:
        Np = threshold
        break

lens_pupil = circ_px(Np, Np_inner)
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
    zernike_poly_combined = combined_zernike(coefficients, npix=Np, N=Np_inner)
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

# normalized k-space frequencies
NA_factor = 1/NA  # custom factor to scale fourier space coordinates such that the max extent correspond to NA=1
FX_norm = FX / (k0 / (2*np.pi)) * NA_factor
FY_norm = FY / (k0 / (2*np.pi)) * NA_factor

# spatial frequency shifts given by the LED positions
# k-space
kxs = LX / np.sqrt(LX ** 2 + LY ** 2 + z0 ** 2) * NA_factor
kys = LY / np.sqrt(LX ** 2 + LY ** 2 + z0 ** 2) * NA_factor

# shifts in pixels units
kxs_px = np.round(kxs*No/2, decimals=0).astype(int)
kys_px = np.round(kys*No/2, decimals=0).astype(int)


# computes the fourier spectrum of my sample
my_sample_FT = fft2c(my_object)

"""
Displays LED matrix, sample, and k-space shifts
"""
if False:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3.5))
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]

    ax1.set_title('LED matrix')
    color_list = [(0.5,0.5,0.5,1)]*nLEDs_x*nLEDs_y
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

    ax3.set_title('Fourier space')
    ax3.pcolormesh(FX_norm, FY_norm, np.log(abs(my_sample_FT) + 0.5), cmap=CMAP_DIFFRACTION)
    # Add a dashed circle to represnet the cutted region by the NA of the lens
    circle_radius = Np_inner / No
    circle = Circle((0, 0), circle_radius, fill=False, linestyle='--', edgecolor='white', linewidth=2, label='NA')
    ax3.add_patch(circle)
    ax3.scatter(kxs, kys, s=10 ,marker='.', c='yellow', label='k-shifts')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.set_xlabel('(NA)')
    ax3.set_ylabel('(NA)')

    fig.tight_layout()
    fig.show()

"""
loops through the k-shifts and extract these regions
then, the recorded image is computed and stored in an array
that array of images we called a ptychogram
"""
ptychogram = np.zeros(shape=(nLEDs_x*nLEDs_y, Np, Np))

plane_wave_simulation = True

if plane_wave_simulation:
    for index, (kxi, kyi) in enumerate(zip(kxs_px.flatten(), kys_px.flatten())):
        print(f'generating frame {index}/{int(nLEDs_x*nLEDs_y)}', end='\r')
        #create slices to select clipped area by the NA in the fourier space
        p1 = slice(int(No / 2 - Np / 2 - kyi), int(No / 2 + Np / 2 - kyi))
        p2 = slice(int(No / 2 - Np / 2 - kxi), int(No / 2 + Np / 2 - kxi))
        # clip Fourier space and apply with the lens pupil that can include aberrations
        my_sample_FT_clipped = my_sample_FT[p1, p2] * lens_pupil
        # FFT of the clipped array and computes the intensity of the field
        # i.e. what the camera sees:
        my_image = fft2c(my_sample_FT_clipped)
        my_detected_image = np.abs(my_image)**2

        # additionally here one can define the noise parameters, photon-count, and bith-depth for discretization
        # of the measured image
        my_detected_image_with_noise = simulate_ccd_image(my_detected_image,
                                                          bit_depth=12,
                                                          peak_photons=10e6,
                                                          quantum_efficiency=0.7,
                                                          quantum_well=None,
                                                          readout_noise=10,
                                                          dc_level=100)
        ptychogram[index, ...] = my_detected_image_with_noise
else:
    for index, (LED_coord_x, LED_coord_y) in enumerate(zip(LX.flatten(), LY.flatten())):
        # evaluate RS integral to compute illumination wavefront that will interact with the sample
        illu_wavefront = RS_point_source_to_plane(LED_coord_x, LED_coord_y, X, Y, z0, wavelength, )
        # Calculate the total energy
        total_energy = np.sum(np.square(np.abs(illu_wavefront)))
        # Normalize the wavefront
        illu_wavefront = illu_wavefront / np.sqrt(total_energy)
        # computes FFT of object*illumination
        my_object_illuminated = my_object * illu_wavefront
        my_sample_FT = fft2c(my_object_illuminated)

        # create slices to select clipped area by the NA in the fourier space
        p1 = slice(int(No / 2 - Np / 2), int(No / 2 + Np / 2))
        p2 = slice(int(No / 2 - Np / 2), int(No / 2 + Np / 2))

        # clip Fourier space and apply with the lens pupil that can include aberrations
        my_sample_FT_clipped = my_sample_FT[p1, p2] * lens_pupil

        # FFT of the clipped array and computes the intensity of the field
        # i.e. what the camera sees:
        my_image = fft2c(my_sample_FT_clipped)
        my_detected_image = np.square(np.abs(my_image))

        # additionally here one can define the noise parameters, photon-count, and bith-depth for discretization
        # of the measured image
        my_detected_image_with_noise = simulate_ccd_image(my_detected_image,
                                                          bit_depth=12,
                                                          peak_photons=10e6,
                                                          quantum_efficiency=0.7,
                                                          quantum_well=None,
                                                          readout_noise=10,
                                                          dc_level=100)

        ptychogram[index, ...] = my_detected_image_with_noise

save_gif = False
if save_gif:
    ptychogram_n = (ptychogram - ptychogram.min()) / (ptychogram.max() - ptychogram.min()) * 255
    ptychogram_n = ptychogram_n.astype(np.uint8)
    imageio.mimsave('ptychogram.gif', ptychogram_n, fps=10, loop=0)


# save data for reconstruciton
day = datetime.date.today().day
month = datetime.date.today().month
year = datetime.date.today().year

# save path
folder = f'datasets/{year}_{month:02}_{day:02}'
os.makedirs(folder, exist_ok=True)

dxd = 6.5e-6  # pixel size of detector
magnification = dxd/dx # magnification, used for FPM computations of dxp
entrancePupilDiameter = 0.5e-3  #entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
encoder = np.stack((kys.flatten()/NA_factor, kxs.flatten()/NA_factor), axis=0)   # diffracted field positions

#show ptychogram
show3Dslider(ptychogram)

with h5py.File(f'{folder}/my_FPM_dataset.h5','w') as hf:
    hf.create_dataset('ptychogram', data=ptychogram)
    hf.create_dataset('wavelength', data=wavelength)
    hf.create_dataset('dxd', data=(dxd,), dtype='f')
    hf.create_dataset('Nd', data=(ptychogram.shape[-1]), dtype='i')
    hf.create_dataset('zled', data=(z0,), dtype='f')
    hf.create_dataset('encoder', data=encoder)
    hf.create_dataset('magnification', data=magnification)
    hf.create_dataset('NA', data=NA)
    hf.create_dataset('entrancePupilDiameter', data=entrancePupilDiameter)
    hf.create_dataset('orientation', data=(0,))

