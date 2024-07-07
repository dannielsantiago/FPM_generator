import matplotlib
try:
    matplotlib.use("tkagg")
except:
    pass
import PtyLab
from PtyLab.io import getExampleDataFolder
from PtyLab import Engines
import logging

logging.basicConfig(level=logging.INFO)


filePath = "datasets/2024_07_07/my_FPM_dataset.h5"

experimentalData, reconstruction, params, monitor, engine, calib = PtyLab.easyInitialize(
    filePath, operationMode="FPM"
)

# experimentalData.magnnification = 4
experimentalData.entrancePupilDiameter = None #entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
experimentalData._setData()
reconstruction.copyAttributesFromExperiment(experimentalData)
reconstruction.computeParameters()
# %% Prepare everything for the reconstruction
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.initialProbe = "circ"
reconstruction.initialObject = "upsampled"

# %% FPM position calibration
# calib.plot = True
# calib.fit_mode ='SimilarityTransform'
# calib.calibrateRadius = True
# calib.fit_mode = "Translation"
# calib.runCalibration()

# %% Prepare reconstruction post-calibration
reconstruction.initializeObjectProbe()

# %% Set monitor properties
monitor.figureUpdateFrequency = 1
monitor.objectPlot = "complex"  # complex abs angle
monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.01  # control object plot FoVW
monitor.probeZoom = 0.01  # control probe plot FoV

# %% Set param
params.gpuSwitch = True
params.positionOrder = "NA"
params.probePowerCorrectionSwitch = False
params.comStabilizationSwitch = False
params.probeBoundary = True
params.adaptiveDenoisingSwitch = True
# Params.positionCorrectionSwitch = True
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

