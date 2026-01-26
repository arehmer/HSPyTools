# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:32:37 2025

@author: rehmer
"""


import numpy as np
import matplotlib.pyplot as plt

from  hspytools.cv import HTPA_Undistorter
from pathlib import Path

plt.close('all')

# %% Load and plot an image to undistort
img_dist_path = Path.cwd() / 'data_samples' / '60x40d_L1k4_0k9_Distortion_TestImage.txt'
img_dist = np.loadtxt(img_dist_path)

vmin = img_dist.min()
vmax = img_dist.max()  

fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Distorted Image')
ax.imshow(img_dist,
          vmin=vmin,
          vmax=vmax)

ax.set_axis_off()

#%% Init HTPA_Undistort
pixel_pitch = 0.045 # mm
w = 60              # pixel
h = 40              # pixel
undistorter = HTPA_Undistorter(w,h,pixel_pitch)

# %% Provide path to grid distortion data
GridDistortionData_path = Path.cwd() / 'data_samples' / 'ZemaxGridDistortion_HTPA60x40d_L1k4_0k9_UHiS.txt' 
undistorter.import_GridDistortionData(GridDistortionData_path)

# %% Estimate mapping to invert distortion
map_x, map_y = undistorter.estimate_mapping()

# %% Apply mapping
img_undist = undistorter.apply_mapping(img_dist, map_x, map_y)

#%% Plot undistorted image
vmin = img_dist.min()
vmax = img_dist.max()  

fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Undistorted Image')
ax.imshow(img_undist,
          vmin=vmin,
          vmax=vmax,)

ax.set_axis_off()
