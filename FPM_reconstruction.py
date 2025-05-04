import h5py
import numpy as np
import matplotlib
try:
    matplotlib.use("tkagg")
except:
    pass
import PtyLab
from PtyLab import Engines
import logging
from PtyLab.utils.utils import fft2c, ifft2c
logging.basicConfig(level=logging.INFO)


filePath = "datasets/2025_05_04/my_FPM_dataset.h5"

experimentalData, reconstruction, params, monitor, engine, calib = PtyLab.easyInitialize(
    filePath, operationMode="FPM"
)

experimentalData.setOrientation(0)
###entrance pupil diamerer
experimentalData.NA=None
# experimentalData.entrancePupilDiameter=20
experimentalData._setData()
reconstruction.copyAttributesFromExperiment(experimentalData)
reconstruction.computeParameters()

with h5py.File(filePath, 'r') as hf:
    positions = hf.get('encoder_px')[()]

# reconstruction.positions0=-1*positions+reconstruction.No//2


# %% Prepare everything for the reconstruction
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.initialProbe = "circ"
reconstruction.initialObject = "upsampled"

# %% FPM position calibration
calib.plot = True
calib.fit_mode ='SimilarityTransform'
calib.calibrateRadius = True
calib.fit_mode = "Translation"
# calib.runCalibration()

# %% Prepare reconstruction post-calibration
reconstruction.initializeObjectProbe()


# %% Set monitor properties
monitor.figureUpdateFrequency = 1
monitor.objectPlot = "complex"  # complex abs angle
monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.1  # control object plot FoVW
monitor.probeZoom = 0.1  # control probe plot FoV

# %% Set param
params.gpuSwitch = True
params.positionOrder = "random"
params.probePowerCorrectionSwitch = False
params.comStabilizationSwitch = False
params.probeBoundary = True
params.adaptiveDenoisingSwitch = True
params.propagator = 'Fourier'
# params.positionCorrectionSwitch = False
# Params.backgroundModeSwitch = True

#%% Run the reconstructors
# Run momentum accelerated reconstructor
engine = Engines.mqNewton(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50
engine.betaProbe = 1
engine.betaObject = 1
engine.beta1 = 0.5
engine.beta2 = 0.5
engine.betaProbe_m = 0.25
engine.betaObject_m = 0.25
engine.momentum_method = "NADAM"
engine.reconstruct()

