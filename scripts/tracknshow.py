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

# # %% Create an argument parser to enable passing argument from the
# # command line to this script
# arg_parser = argparse.ArgumentParser(description="Script with keyword arguments.")

# # %% Add arguments using '--key' style
# arg_parser.add_argument("--w", 
#                         type=int,
#                         required=True,
#                         metavar="width",
#                         help="Width of sensor array in pixels")

# arg_parser.add_argument("--h",
#                         type=int,
#                         required=True,
#                         metavar="height",
#                         help="Height of sensor array in pixels")

# arg_parser.add_argument("--id",
#                         type=str,
#                         required=True,
#                         metavar="DevID",
#                         help="IP of HTPA-Appset to bind")

# # arg_parser.add_argument("--tracktor_dict",
# #                         type=str,
# #                         required=True,
# #                         metavar="dict",
# #                         help="Path to dictionary with tracktor parameters")

# # %% Parse arguments
# args = arg_parser.parse_args()

# w = args.w
# h = args.h
# devid = args.id


# %%
w = 60
h = 40
devid = 3712
bcast_addr = '192.168.240.255'
# tracktor_dict 
# tracktor_dict = Path(args.tracktor_dict)

# %% Only for debugging!
# w =  60
# h = 40
# ip = '192.168.240.160' 
# tracktor_dict = 'H:/PersonDetectionModels/160x120_L3k95_standing_ang0/training/model_0/160x120_L3k95_standing_ang0_balanced.tracktor'
# tracktor_dict = 'H:/PersonDetectionModels/60x40_standing_angle0/training/model_3/60x40_L1k9_standing_ang0.tracktor'

# %% Initialize tracktor
# tracktor_dict = pkl.load(open(tracktor_dict,'rb'))

# tracktor = build_tracktor(**tracktor_dict)

# %% Create instance of UDP reader
udp_reader = HTPA_UDPReader(w,h)

# %% Code begins here
udp_buffer = Queue(maxsize=1)

# Create buffers for communication between threads
udp_buffer = Queue(maxsize=1)
detect_buffer = Queue(maxsize=1)

# Create condition variables for thread synchronization
udp_buffer_lock = Condition()
detect_buffer_lock = Condition()
writer_lock = Condition()

# Create instance of UDP thread
udp_thread = UDP(udp_reader = udp_reader,
                 # IP = ip,
                 DevID = devid,
                 Bcast_Addr = bcast_addr,
                 write_buffer = udp_buffer,
                 write_condition = udp_buffer_lock)

plot_thread = Imshow(width = w,
                     height = h,
                     read_buffer = udp_buffer,
                     read_condition = udp_buffer_lock)

# detection_thread = Tracktor_Thread(width = w,
#                                    height = h,
#                                    tracktor = tracktor,
#                                    write_buffer = detect_buffer,
#                                    write_condition = writer_lock,
#                                    read_buffer = udp_buffer,
#                                    read_condition = udp_buffer_lock)

# plot_thread = Imshow(width = w,
#                      height = h,
#                      read_buffer = detect_buffer,
#                      read_condition = writer_lock)

if __name__ == '__main__':
    
      
    udp_thread.start()
    # detection_thread.start()
    plot_thread.start()
    
    # Sleep a few seconds
    # time.sleep(45)
    
    # # Stop the threads in reverse order!
    # plot_thread.stop()
    # detection_thread.stop()
    # udp_thread.stop()
