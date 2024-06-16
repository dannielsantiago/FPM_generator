import numpy as np
import matplotlib.pyplot as plt
import h5py
from IAP.Tools.misc import *
import matplotlib

matplotlib.use('Qt5Agg')

def load_file(folder, file):
    with h5py.File(folder+file,'r') as f:
        object_ = np.squeeze(f.get('object')[()])
        probe_ = np.squeeze(f.get('probe')[()])
        dx = f.get('dxp')[()]
        wl = f.get('wavelength')[()]
    return object_, probe_, dx, wl


#%%
#load files
folder = '../Datasets/Experimental_data/reconstructed/LED/16 Beams/'
files = {'84%_1': '20072023_100us_84_overlap_400pos_binned_2npsm_FRC.hdf5',
         '84%_2': '20072023_100us_84_overlap_400pos_binned_2_2npsm_FRC.hdf5',
         '84%_a': '20072023_100us_84_overlap_400pos_binned_2npsm_bkcorrected_FRC.hdf5',
         '84%_b': '20072023_100us_84_overlap_400pos_binned_2npsm_bkfiltered_FRC.hdf5',
         '84%_ab': '20072023_100us_84_overlap_400pos_binned_2npsm_bkcorrected_filtered_FRC.hdf5',
         '77%_1': '20072023_100us_77_overlap_200pos_binned_2npsm_FRC.hdf5',
         '77%_2': '20072023_100us_77_overlap_200pos_binned_2_2npsm_FRC.hdf5',
         '68%_1': '20072023_100us_68_overlap_100pos_binned_2npsm_FRC.hdf5',
         '68%_2': '20072023_100us_68_overlap_100pos_binned_2_2npsm_FRC.hdf5',
         'LFOV_1': '20072023_LFOV_100us_68_overlap_400pos_binned_2npsm_FRC.hdf5',
         'LFOV_2': '20072023_LFOV_100us_68_overlap_400pos_binned_2_2npsm_FRC.hdf5',
         'SingleObj': '20072023_100us_84_overlap_400pos_binned_singleObj_32beams.hdf5',
         'SingleObj_LFOV': '20072023_100us_84_overlap_400pos_binned_singleObj_32beams.hdf5'}

object_1, probe_1, dx, wl = load_file(folder,files['84%_2'])
object_1 = object_1[5]
object_2, _, _, _ = load_file(folder,files['84%_b'])
object_2 = object_2[5]

folder = '../Datasets/Experimental_data/reconstructed/LED/4 Beams/'
files = {'84%_1': '24072023_4Beams_100us_84_overlap_400pos_binned_4npsm_FRC.hdf5',
         '84%_2': '24072023_4Beams_100us_84_overlap_400pos_binned_2_4npsm_FRC.hdf5',
         '77%_1': '24072023_4Beams_100us_77_overlap_200pos_binned_4npsm_FRC.hdf5',
         '77%_2': '24072023_4Beams_100us_77_overlap_200pos_binned_2_4npsm_FRC.hdf5',
         '68%_1': '24072023_4Beams_100us_68_overlap_100pos_binned_4npsm_FRC.hdf5',
         '68%_2': '24072023_4Beams_100us_68_overlap_100pos_binned_2_4npsm_FRC.hdf5',}

# object_1, probe_1, dx, wl = load_file(folder,files['68%_1'])
# object_1 = object_1[0]
# object_2, _, _, _ = load_file(folder,files['68%_2'])
# object_2 = object_2[0]

folder = '../Datasets/Experimental_data/reconstructed/LED/1 Beam/'
files = {'84%_1': '24072023_1Beam_100us_84_overlap_400pos_binned_2npsm_FRC.hdf5',
         '84%_2': '24072023_1Beam_100us_84_overlap_400pos_binned_2_2npsm_FRC.hdf5',
         '77%_1': '24072023_1Beam_100us_77_overlap_200pos_binned_2npsm_FRC.hdf5',
         '77%_2': '24072023_1Beam_100us_77_overlap_200pos_binned_2_2npsm_FRC.hdf5',
         '68%_1': '24072023_1Beam_100us_68_overlap_100pos_binned_2npsm_FRC.hdf5',
         '68%_2': '24072023_1Beam_100us_68_overlap_100pos_binned_2_2npsm_FRC.hdf5',}


# object_1, probe_1, dx, wl = load_file(folder,files['84%_1'])
# object_2, _, _, _ = load_file(folder,files['68%_1'])

#%%


myFRC = MyFRC(object_1, object_2, dx)
# myFRC.show_raw_data()
myFRC.normalize_amplitude()
myFRC.remove_phase_ramp()
myFRC.show_comparison_after_phase_ramp_removal(id=0)
myFRC.show_comparison_after_phase_ramp_removal(id=1)
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

