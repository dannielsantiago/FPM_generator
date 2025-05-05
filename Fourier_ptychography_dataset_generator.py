import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5agg")
from PtyLab.utils.visualisation import show3Dslider
from Tools.propagators import fft2c, ifft2c
from Tools.misc import *
from Tools.zernike_polynomials import *
from matplotlib.patches import Circle
from Tools.multiprocessing_scripts import RS_diffraction_integral, RS_point_source_to_plane
import imageio.v2 as imageio
import datetime
import os
import h5py
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')  # Transpose row-col for display plots
import gc


"""
Load image as sample
"""
# Load an image and normalize it
my_object_RGB = plt.imread('imgs/TCGA.jpg')  # RGB image
# normalize its amplitude
my_object_amp = np.mean(my_object_RGB, axis=-1)
my_object_amp /= np.amax(my_object_amp)

# adds phase information
max_phase_variation = np.pi
stepness=10
my_object_phase = abs(1 - my_object_amp)
my_object_phase = max_phase_variation * 0.5 * (1 + np.tanh(stepness * (my_object_phase - 0.5)))
# my_object_phase = np.clip(my_object_phase, 0, max_phase_variation)

# constructs complex-valued object
my_object = my_object_amp*np.exp(-1j*my_object_phase)
# Keep the dimention of my_object to 2048x2048. To have the same reference for all simulations
#clean variables
del my_object_phase
del my_object_amp
del my_object_RGB
gc.collect()

"""
Define experimental parameters
"""
NA = 0.0039*2 #  # detection NA: 0.0039-->32x32, 0.00781-->64x64, 0.0156-->128x128, 0.03125-->256x256, 0.0625-->512x512, 0.125-->1024x1024
#Define my illumiation NA
#this should be larger than detection NA
#to have comparable results, each simulation will try to retrieve a 3x larger NA
NA_illu = 10*NA
#defines how many LEDs we want to use. A lower number of LEDs will result in a lower overlap.
# we need to investigate what is the min. number of LEDs to have a good reconstuction for a given combination of
# NA_illu and NA detection
#usar numeros impares para que siempre haya un LED en la posicion 0,0
nLEDs_x = 9
nLEDs_y = 9

#Define the lateral size of my LED matrix
LM = 0.1 # let's say 10cm, adjust it to the experimental one
#computes the needed z0 to obtain the desired NA_illu
z0 = LM/(2*NA_illu)
#computes the led separation distance
dlx = LM/(nLEDs_x-1)  # Led separation distance along x direction
dly = LM/(nLEDs_y-1) #led separation distance along y direction
wavelength = 450e-9  #LED wavelength illumination
k0 = 2*np.pi/wavelength  #wavenumber
#additional parameters, not so relevant for simulations, but for experimental datasets
magnification = 2
dxd = 5.5e-6  # pixel size of detector
dx = dxd / magnification  # pixel size defined by the magnification of the objective
No = my_object.shape[-1]  # number of pixels of my object - Asumming square object
# create lens pupil
Np_inner = int(2 * NA * No) #Diameter in pixels
# List of threshold values. This ensures that the final images are power of 2
thresholds = [32, 64, 128, 256, 512, 1024, 2048,4096]
# Calculate Np based on the value of No and NA
# Find the next threshold value greater than or equal to Np
for threshold in thresholds:
    if 2*Np_inner < threshold:
        Np = threshold
        break

lens_pupil = circ_px(Np, Np_inner)
#smooth the edges of the pupil via convolution
lens_pupil = np.real(ifft2c(fft2c(lens_pupil) * fft2c(circ_px(Np, int(Np*0.2)))))**4  # smooth edges by convolution
lens_pupil /=np.amax(lens_pupil)


# optional, add aberrations to the lens pupil
add_aberrations = False
if add_aberrations:
    # Define Zernike coefficients (m, n, coefficient)
    coefficients = [
        (0, 0, 0),  # Piston
        (1, 1, 0),  # Tilt X
        (1, -1, 0),  # Tilt Y
        (0, 2, 0.25),  # Defocus
        (2, 2, 0),  # Astigmatism 45°
        (2, -2, 0),  # Astigmatism 0°
    ]
    # Generate the combined Zernike polynomial
    zernike_poly_combined = combined_zernike(coefficients, npix=Np, N=Np)

    phase_aberration = zernike_poly_combined
    # Complex transmission function
    lens_pupil = lens_pupil*np.exp(1j * phase_aberration)


#creates 2d-arrays for the positions of each LED
L_led_x = (nLEDs_x-1) * dlx  # lateral extension of led matrix
L_led_y = (nLEDs_y-1) * dly  # lateral extension of led matrix
lx = np.linspace(-L_led_x/2, L_led_x/2, nLEDs_x)
ly = np.linspace(-L_led_y/2, L_led_y/2, nLEDs_y)
LX, LY = np.meshgrid(lx, ly)  # 2d- grid coordinates

# LED_color = wavelength_to_rgb(wavelength*1e9)
# LED_color_normalized = [x/255 for x in LED_color]  # Converted to 0-1 range, with alpha=1.0

# spatial frequency shifts given by the LED positions
# k-space
kxs = LX / np.sqrt(LX ** 2 + LY ** 2 + z0 ** 2)
kys = LY / np.sqrt(LX ** 2 + LY ** 2 + z0 ** 2)

# shifts in pixels units
kxs_px = np.round(kxs*No/2, decimals=0).astype(int)
kys_px = np.round(kys*No/2, decimals=0).astype(int)

# computes the fourier spectrum of my sample
my_sample_FT = fft2c(my_object)

"""
Displays LED matrix, sample, and k-space shifts
"""
if False:
    # sample coordinates
    L = No * dx  # sample's lateral size in meters

    # real space coordinates of sample
    x = np.arange(-No / 2, No / 2) * dx
    X, Y = np.meshgrid(x, x)

    # fourier space coordinates of sample
    f = np.arange(-No / 2, No / 2) / L
    FX, FY = np.meshgrid(f, f)
    # normalized k-space frequencies
    NA_factor = 1 / NA  # custom factor to scale fourier space coordinates such that the max extent correspond to NA=1
    FX_norm = FX / np.amax(FX)
    FY_norm = FY / np.amax(FX)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,3.5))
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    ax4 = ax[3]

    ax4.set_title('Commplex Pupil')
    ax4.imshow(complex2rgb(lens_pupil))
    ax4.set_axis_off()

    ax1.set_title('LED matrix')
    color_list = [(0.5,0.5,0.5,1)]*nLEDs_x*nLEDs_y
    ax1.scatter(LX*1e2, LY*1e2, marker='s', c=color_list)
    ax1.set_xlabel('(cm)')
    ax1.set_ylabel('(cm)')
    ax1.set_aspect('equal')
    ax1.minorticks_on()
    ax1.grid(True, alpha=0.5)

    ax2.set_title('Complex-valued object')
    # ax2.pcolormesh(X * 1e3, Y * 1e3, np.ones(shape=(No, No)), color=complex2rgb(my_object).reshape(-1, 3) / 255)
    # add_complex_colorwheel(fig, ax2, loc=4, pad=0.02)
    ax2.imshow(complex2rgb(my_object))
    ax2.set_aspect('equal')
    ax2.set_xlabel('(mm)')
    ax2.set_ylabel('(mm)')

    ax3.set_title('Fourier space')
    ax3.pcolormesh(FX_norm, FY_norm, np.log(abs(my_sample_FT) + 0.5), cmap=CMAP_DIFFRACTION)
    # Add a dashed circle to represnet the cutted region by the NA of the lens
    circle_radius = 0.5 * Np_inner / No
    #adds circles to each center point in Fourier space
    for i, (kx_i, ky_i) in enumerate(zip(kxs.flatten(), kys.flatten())):
        temp_circle = Circle((kx_i, ky_i), circle_radius, fill=False, linestyle='-', edgecolor='white', linewidth=1)
        ax3.add_patch(temp_circle)

    circle = Circle((0, 0), circle_radius, fill=False, linestyle='--', edgecolor='red', linewidth=1, label='NA')
    ax3.add_patch(circle)

    ax3.scatter(kxs, kys, s=10 ,marker='.', c='yellow', label='k-shifts')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.set_xlabel('(NA)')
    ax3.set_ylabel('(NA)')

    fig.tight_layout()
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    #clean varialbes from memory
    # del FX
    # del FX_norm
    # del FY
    # del FY_norm
    # del X
    # del Y
    # gc.collect()

"""
loops through the k-shifts and extract these regions
then, the recorded image is computed and stored in an array
that array of images we called a ptychogram
"""
ptychogram = np.zeros(shape=(nLEDs_x*nLEDs_y, Np, Np))

spiral_order = True
if spiral_order:
    s_indices = list(spiral_indices(nLEDs_y, nLEDs_x))
    # Option 1: Create new 1D arrays in spiral order to store positions of LEDs as encoder
    LX = np.array([LX[i, j] for i, j in s_indices])
    LY = np.array([LY[i, j] for i, j in s_indices])
    encoder = np.stack((LY, LX), axis=-1)   # diffracted field positions
    kxs_px = np.array([kxs_px[i, j] for i, j in s_indices])
    kys_px = np.array([kys_px[i, j] for i, j in s_indices])
else:
    #default row major sequential order of positions
    encoder = np.stack((LY.flatten(), LX.flatten()), axis=-1)   # diffracted field positions

conv = (dx * Np / wavelength)
# encoder_px = np.stack((kys_px, kxs_px), axis=-1)
encoder_px2 = np.round( conv * encoder / np.sqrt(encoder[:,0] ** 2 + encoder[:,1] ** 2 + z0**2)[..., None])

#Calculates overlap between detected fourier patches
diffs = np.diff(encoder_px2, axis=0)
distances = np.hypot(diffs[:, 0], diffs[:, 1])
average_distance = distances.mean() # average distance

overlap_linear = linear_overlap(Np_inner, average_distance)
overlap_area = area_overlap(Np_inner, average_distance)
print(f'linear overlap: {overlap_linear*100:.2F}%')
print(f'area overlap: {overlap_area*100:.2F}%')

plane_wave_simulation = True
if plane_wave_simulation:
    # for index, (kxi, kyi) in enumerate(zip(kxs_px.flatten(), kys_px.flatten())):
    for index, (kyi, kxi) in enumerate(encoder_px2):
        print(f'generating frame {index}/{int(nLEDs_x * nLEDs_y)}', end='\r')
        #create slices to select clipped area by the NA in the fourier space
        p1 = slice(int(No / 2 - Np / 2 - kyi), int(No / 2 + Np / 2 - kyi))
        p2 = slice(int(No / 2 - Np / 2 - kxi), int(No / 2 + Np / 2 - kxi))
        # clip Fourier space and apply with the lens pupil that can include aberrations
        # my_sample_FT_clipped = my_sample_FT[p1, p2] * lens_pupil
        my_sample_FT_clipped = clip_my_sample_FT(my_sample_FT, p1, p2) * lens_pupil
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
                                                          dc_level=0)
        # ptychogram[index, ...] = my_detected_image_with_noise
        ptychogram[index, ...] = my_detected_image#_with_noise

else:
    # create slices to select clipped area by the NA in the fourier space
    p1 = slice(int(No / 2 - Np / 2), int(No / 2 + Np / 2))
    p2 = slice(int(No / 2 - Np / 2), int(No / 2 + Np / 2))

    for index, (LED_coord_x, LED_coord_y) in enumerate(zip(LX.flatten(), LY.flatten())):
        print(f'generating frame {index}/{int(nLEDs_x * nLEDs_y)}', end='\r')
        # evaluate RS integral to compute illumination wavefront that will interact with the sample
        illu_wavefront = RS_point_source_to_plane(LED_coord_x, LED_coord_y, X, Y, z0, wavelength, )
        # Calculate the total energy
        # total_energy = np.sum(np.square(np.abs(illu_wavefront)))
        # # Normalize the wavefront
        # illu_wavefront = illu_wavefront / np.sqrt(total_energy)
        # computes FFT of object*illumination
        my_object_illuminated = my_object * illu_wavefront
        my_sample_FT = fft2c(my_object_illuminated)

        my_sample_FT_clipped = my_sample_FT[p1, p2]
        # clip Fourier space and apply with the lens pupil that can include aberrations
        my_sample_FT_clipped = my_sample_FT_clipped * lens_pupil

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
                                                          dc_level=0)

        ptychogram[index, ...] = my_detected_image#_with_noise

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

# entrancePupilDiameter = 2*NA*No*dx
# entrancePupilDiameter = 2 * Np * NA * dx**2 / wavelength

#computes the goal reference image for the given simulation parameters
height = int(np.max(encoder_px2[:,0]) - np.min(encoder_px2[:,0])) + Np
width = int(np.max(encoder_px2[:,1]) - np.min(encoder_px2[:,1])) + Np
p1 = slice(int(No / 2  - height/2), int(No / 2 + height/2))
p2 = slice(int(No / 2  - width/2), int(No / 2 + width/2))

my_target_image = fft2c(my_sample_FT[p1, p2])
my_target_amp = abs(my_target_image)
folder2 = f'{folder}/{Np}x{Np}_dataset'
os.makedirs(folder2, exist_ok=True)
plt.imsave(f'{folder2}/target_amplitude_reference_{height}x{width}_NA_illu_{NA_illu}.png', my_target_amp, cmap='gray')
plt.imsave(f'{folder2}/target_complex_reference_{height}x{width}_NA_illu_{NA_illu}.png', complex2rgb(my_target_image))

filename = f'my_FPM_dataset_{Np}x{Np}_overlap_{overlap_linear*100:.2F}.h5'
with h5py.File(f'{folder2}/{filename}','w') as hf:
    hf.create_dataset('ptychogram', data=ptychogram)
    hf.create_dataset('wavelength', data=wavelength)
    hf.create_dataset('dxd', data=(dxd,), dtype='f')
    hf.create_dataset('Nd', data=(ptychogram.shape[-1]), dtype='i')
    hf.create_dataset('zled', data=(z0,), dtype='f')
    hf.create_dataset('encoder', data=encoder)
    hf.create_dataset('magnification', data=magnification)
    # hf.create_dataset('NA', data=NA)
    # hf.create_dataset('entrancePupilDiameter', data=entrancePupilDiameter)
    hf.create_dataset('orientation', data=(0,))

print(f'file saved in {folder2}/{filename}')


#show ptychogram. This is the input dataset that will be used to reconstruct a larger image
# show3Dslider(ptychogram)


#clean variables and memory
# del ptychogram
# del my_object
# del my_sample_FT
# gc.collect()

