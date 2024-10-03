import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Tools.misc import complex2rgb, fft2c, ifft2c
try:
    matplotlib.use("QtAgg")

    # matplotlib.use("tkagg")
except:
    pass
import PtyLab
from PtyLab import Engines
from PtyLab.utils.visualisation import modeTile, complex2rgb
import logging
import copy

logging.basicConfig(level=logging.INFO)


filePath = "datasets/Thomas_Aidukas/USAFTargetFPM.hdf5"
filePath = "datasets/Thomas_Aidukas/LungCarcinomaFPM.hdf5"


experimentalData, reconstruction, params, monitor, engine, calib = PtyLab.easyInitialize(
    filePath, operationMode="FPM"
)

mean_img = np.mean(experimentalData.ptychogram,0)
experimentalData.showptychogram()
# experimentalData.magnnification = 4
# experimentalData.entrancePupilDiameter = None #entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
# experimentalData._setData()
# reconstruction.copyAttributesFromExperiment(experimentalData)
# reconstruction.computeParameters()
# %% Prepare everything for the reconstruction
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
reconstruction.initialProbe = 'circ'
reconstruction.initialObject = 'upsampled'
reconstruction.initializeObjectProbe()

# Set monitor properties
monitor.figureUpdateFrequency = 10
monitor.objectPlot = 'complex'
monitor.verboseLevel = 'low'
monitor.objectPlotZoom = .01
monitor.probePlotZoom = .01

#params class
params.gpuSwitch = True
params.positionOrder = 'NA'
params.probeBoundary = True
params.adaptiveDenoisingSwitch = True

# %% FPM position calibration
calib.plot = True
calib.fit_mode ='SimilarityTransform'
calib.calibrateRadius = True
calib.runCalibration()

#%% Run the reconstructors
# Run momentum accelerated reconstructor
engine = Engines.mqNewton(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50
# engine.betaProbe = 1
# engine.betaObject = 1
# engine.beta1 = 0.5
# engine.beta2 = 0.5
# engine.betaProbe_m = 0.25
# engine.betaObject_m = 0.25
# engine.momentum_method = "NADAM"
engine.reconstruct()


plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams['figure.dpi'] = 100

fig = plt.figure()
plt.subplot(131)
plt.title('raw data')
plt.imshow(abs(mean_img),cmap='gray')
plt.subplot(132)
plt.title('reconstructed amplitude')
plt.imshow(abs(modeTile(fft2c(reconstruction.object))),cmap='gray')
plt.subplot(133)
plt.title('complex reconstruction')
plt.imshow(complex2rgb(modeTile(fft2c(reconstruction.object))))
plt.show()
fig.canvas.draw()
fig.canvas.flush_events()

#Remove phase ramp
wavelength = reconstruction.wavelength
Np = reconstruction.Np
No = reconstruction.No
dxp = reconstruction.dxp
z=120e-3 # illumination to sample distance
u=60e-3 # sample to lens distance
# create the coordinate grid
x = np.linspace(-Np / 2 , Np / 2, No)
Yp, Xp = np.meshgrid(x, x)
Yp = Yp*dxp
Xp = Xp*dxp

# illumination phase curvature
ill_curvature = np.exp(1j*np.pi/wavelength * (1/z) * (Xp**2 + Yp**2))
# non-telecentricity induced phase curvature
tele_curvature = np.exp(1j*np.pi/wavelength * (1/u) * (Xp**2 + Yp**2))
# combined curvature
total_curvature = ill_curvature * tele_curvature

# plt.figure()
# plt.subplot(132)
# plt.title('Phase curvature')
# plt.imshow(complex2rgb(total_curvature))
# plt.show()

# Initialize the object + curvature
reconstruction = copy.deepcopy(reconstruction)
reconstruction.object = ifft2c(fft2c(reconstruction.object) * total_curvature.conj())

# Reconstruct
engine = Engines.qNewton(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50
engine.reconstruct()

# Remove the curvature
fixed_object = modeTile(fft2c(reconstruction.object) * total_curvature)

fig = plt.figure()
plt.subplot(131)
plt.title('raw data')
plt.imshow(abs(mean_img),cmap='gray')
plt.subplot(132)
plt.title('reconstructed amplitude')
plt.imshow(abs(fixed_object),cmap='gray')
plt.subplot(133)
plt.title('complex reconstruction')
plt.imshow(complex2rgb(fixed_object))
plt.show()
fig.canvas.draw()
fig.canvas.flush_events()
# engine.reconstruct()
