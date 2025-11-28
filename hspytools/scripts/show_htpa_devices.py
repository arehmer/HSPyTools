# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""

import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray
from hspytools.ipc.threads import UDP,Imshow

from hspytools.readers import HTPA_UDPReader
from hspytools.ipc.threads import UDP, Record_Thread, FileWriter_Thread
from hsod.ipc.threads import Tracktor_Thread
from hsod.cv.tracktors import build_tracktor

from threading import Condition

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(description="Script with keyword arguments.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--w", 
                        type=int,
                        required=True,
                        metavar="width",
                        help="Width of sensor array in pixels")

arg_parser.add_argument("--h",
                        type=int,
                        required=True,
                        metavar="height",
                        help="Height of sensor array in pixels")


arg_parser.add_argument("--bcast",
                        type=str,
                        required=True,
                        metavar="bcast",
                        help="Broadcast Address to look for HTPA device")




# %% Parse arguments
args = arg_parser.parse_args()
w = args.w
h = args.h
bcast = args.bcast


# %% Create instance of UDP reader
udp_reader = HTPA_UDPReader(w,h)

# %% Broadcast looking for devices
devices = udp_reader.broadcast(bcast)


print(devices)
