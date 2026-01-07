# -*- coding: utf-8 -*-
"""
HTPA Device Discovery and Binding Script
----------------------------------------


This script discovers HTPA thermal imaging devices on a network using a user-
provided broadcast address. After detecting devices via a UDP broadcast, the
script interactively prompts the user to select one of the discovered devices.
It then initializes the appropriate ArrayType for that device, starts threads
responsible for receiving UDP packets and visualizing the frames using OpenCV,
keeps them running for approximately 20 seconds, and finally shuts them down
cleanly.


Usage Examples
==============


Windows PowerShell:
python ./StreamHTPADevice.py --bcast 192.168.178.255


Linux shell (bash):
python ./StreamHTPADevice.py --bcast 192.168.178.255


Command-line Arguments
======================
--bcast : str (required)
Broadcast address used to detect HTPA devices on the network.


This script follows standard Python CLI practices, using argparse for argument
parsing and conventional object/thread lifecycle management.
"""


import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray
from hspytools.ipc.threads import UDP,Imshow

# from hspytools.ipc.threads import UDP, Record_Thread, FileWriter_Thread

from threading import Condition

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(description="Script with keyword arguments.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bcast",
                        type=str,
                        required=True,
                        metavar="bcast",
                        help="Broadcast Address to look for HTPA device")


# %% Parse arguments
args = arg_parser.parse_args()
bcast_addr = args.bcast

# %% Main loop
if __name__ == '__main__':
    
    # %% Create instance of UDP reader
    udp_reader = HTPA_UDPReader()
    
    # %% Broadcast looking for devices
    devices = udp_reader.broadcast(bcast_addr)
    
    
    # Get user input on which device to bind
    while True:
        try:
            DevID = int(input('Enter the device number of the HTPA device to bind: ' ))
            if DevID in udp_reader.devices.index:
                break
            else:
                print(f'HTPA device with ID {DevID} not discovered.')
        except ValueError:
            print("Please enter a valid integer.")
    
    
    # Get the ArrayType of the selected device
    ArrayType = devices.loc[DevID,'Arraytype']
    
    # Set the ArrayType of the HTPA_UDPReader class
    udp_reader.ArrayType = ArrayType
    
    # Create threads which bind the device, receive the udp packages and 
    # plot the frame
    

    # Create buffers for communication between threads
    udp_buffer = Queue(maxsize=1)

    # Create condition variables for thread synchronization
    udp_buffer_lock = Condition()

    # Create instance of UDP thread 
    udp_thread = UDP(udp_reader = udp_reader,
                     DevID = DevID,
                     Bcast_Addr = bcast_addr,
                     write_buffer = udp_buffer,
                     write_condition = udp_buffer_lock)
    
    # Create instance of Imshow thread for plotting using cv2.imshow
    plot_thread = Imshow(ArrayType = ArrayType,
                         read_buffer = udp_buffer,
                         read_condition = udp_buffer_lock)
    
    
    # Start the threads
    udp_thread.start()
    plot_thread.start()

    # Let threads run 20 seconds
    time.sleep(20)
    
    # Stop the threads in reversed order!
    plot_thread.stop()
    udp_thread.stop()