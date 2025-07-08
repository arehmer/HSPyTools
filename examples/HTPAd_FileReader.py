# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:56:46 2024

@author: rehmer
"""

from hspytools.readers import HTPAdGUI_FileReader
from hspytools.tparray import TPArray, SensorTypes


from pathlib import Path

# %% path to file as Path object
file_path = Path.cwd() / 'data_samples' / '60x40L1k9.BDS'

# %% All implemented sensor types are in the dictionary SensorTypes
print(SensorTypes.keys())

#%% Initialize a TPArray instance providing the proper Sensor Type
tparray = TPArray(SensorType = SensorTypes['HTPA60x40D_L1K9_0K8'])

# %% Initialize File Reader using the TPArray instance
reader = HTPAdGUI_FileReader(tparray)

# %% Read in bds as pandas DataFrame using read_htpa_video() method
# read_htpa_video() returns the content of the bds as dataframe as well as
# the original header of the bds file 
df,header = reader.read_htpa_video(file_path)


# %% How to access data in the DataFrame?

# tparray._pix returns all columns that contain pixel values
df_pixel = df[tparray._pix]


# %% Get a single frame (frame 10)
frame10 = df.loc[10,tparray._pix]

# %% Plot frame
import matplotlib.pyplot as plt
frame10_np = frame10.values.reshape(tparray._npsize)

plt.figure('Frame 10')
plt.imshow(frame10_np)


print("The origin in the .bds or .txt file is in the upper right " +\
        "corner. matplotlib hat its origin in the upper left corner. " +\
        "I.e. the image needs to be flipped on the vertical axis for " +\
        "proper display.")
    
import numpy as np
frame10_flipped = np.flip(frame10_np,axis=1)

plt.figure('Frame 10 flipped')
plt.imshow(frame10_flipped)




