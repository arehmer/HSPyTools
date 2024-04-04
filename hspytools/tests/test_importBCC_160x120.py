# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:07:54 2024

@author: Rehmer
"""

# import pandas as pd
from pathlib import Path
# import numpy as np
# import h5py
import matplotlib.pyplot as plt
import pickle as pkl
from hsfit.cmap_mgr import cmap_mgr

# # %% Test implementation
from hspytools.readers import HTPAdGUI_FileReader
from hspytools.tparray import TPArray

# %% Specify paths
bcc_47 = Path.cwd() / 'data'  / 'bcc' / '02042024_ 132630____47.BCC'

# meas_path = Path.cwd() / 'data' / 'LaudaBB' / 'meas' 

# Toref_path = Path.cwd() / 'data' / 'LaudaBB' / 'ref' / 'Versuchsplan_160x120_L10k0_F0k72_LaudaBB.csv'

# cmap_path = Path.cwd() / 'cmap_160x120_L10k0_LaudaBB.pkl'

# %% 

tparray = TPArray(160,120)

BCC = tparray.import_BCC(bcc_47)