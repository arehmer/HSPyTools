# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:07:54 2024

@author: Rehmer
"""


from pathlib import Path

from hspytools.tparray import TPArray
import matplotlib.pyplot as plt

# %% Specify paths
bcc_1234 = Path.cwd() / 'BCCs'  / '120x84_L3k9 1234.BCC'



# %% 

tparray = TPArray(120,84)

BCC = tparray.import_BCC(bcc_1234)

# %%

plt.imshow(BCC['pij'])
