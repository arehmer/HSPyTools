# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:07:54 2024

@author: Rehmer
"""


from pathlib import Path

from hspytools.tparray import TPArray

# %% Specify paths
bcc_47 = Path.cwd() / 'BCCs'  / '160x120_L10k0_F0k72_ID47.BCC'



# %% 

tparray = TPArray(160,120)

BCC = tparray.import_BCC(bcc_47)