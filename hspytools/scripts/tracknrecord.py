# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""


# import numpy as np
import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

import signal
import sys

import logging
import threading

from hspytools.readers import HTPA_UDPReader
from hspytools.ipc.threads import UDP
from threading import Condition
from wwnpi.threads import NextcloudUpload_Thread, MotionDetectorThread, ExtendedRecordThread
from wwnpi.detector import MotionDetector


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

arg_parser.add_argument("--devid",
                        type=int,
                        required=True,
                        metavar="devid",
                        help="DeviceID of HTPA-device to bind")

arg_parser.add_argument("--bcast",
                        type=str,
                        required=True,
                        metavar="bcast",
                        help="Broadcast Address of subnet in which to search for HTPA devices")


arg_parser.add_argument("--save_dir",
                        type=str,
                        required=True,
                        metavar="folder",
                        help="Path to folder for storing detection results")

# %% Parse arguments
# Parse arguments
args = arg_parser.parse_args()

w = args.w
h = args.h
devid = args.devid
bcast_addr = args.bcast
save_dir = Path(args.save_dir)

# %% Only for debugging!
UDPPorts = {4658:30444,
            4684:30445,
            3714:30446,
            4652:30447,
            4656:30448,
            3716:30449}

# %% Initialize a motion detector
detector = MotionDetector()

# %% Create instance of UDP reader
udp_reader = HTPA_UDPReader(w,h,port=UDPPorts[devid])

# %% Initialize buffers and threads

# Create buffers for communication between threads
udp_buffer = Queue(maxsize=1)
detect_buffer = Queue(maxsize=1)
upload_buffer = Queue()

# Create condition variables for thread synchronization
udp_buffer_lock = Condition()
detect_buffer_lock = Condition()
upload_lock = Condition()

udp_thread = UDP(udp_reader = udp_reader,
                 DevID = devid,
                 Bcast_Addr = bcast_addr,
                 write_buffer = udp_buffer,
                 write_condition = udp_buffer_lock)

detection_thread = MotionDetectorThread(width = w,
                                        height = h,
                                        detector = detector,
                                        write_buffer = detect_buffer,
                                        write_condition = detect_buffer_lock,
                                        read_buffer = udp_buffer,
                                        read_condition = udp_buffer_lock)

record_thread = ExtendedRecordThread(width = w,
                                     height = h,
                                     read_buffer = detect_buffer,
                                     read_condition = detect_buffer_lock,
                                     write_buffer = upload_buffer,
                                     write_condition = upload_lock,
                                     n_pre_record = 120,
                                     save_dir = save_dir,
                                     imshow = True,
                                     recording_timeout=60)  # Optional: 2-minute timeout

upload_thread = NextcloudUpload_Thread(read_buffer = upload_buffer,
                                        read_condition = upload_lock)

# %% Define a function which can shut down threads on request
def signal_handler(sig, frame):
    print("Gracefully stopping threads...")
    upload_thread.stop()
    record_thread.stop()
    detection_thread.stop()
    udp_thread.stop()
    sys.exit(0)

# %% Define a function that monitors the status of the threads
def monitor_threads():
    while True:
        time.sleep(60)
        if not udp_thread.is_alive():
            logging.error("UDP thread stopped unexpectedly!")
        if not detection_thread.is_alive():
            logging.error("Detection thread stopped unexpectedly!")
        if not record_thread.is_alive():
            logging.error("Recording thread stopped unexpectedly!")
        if not upload_thread.is_alive():
            logging.error("Upload thread stopped unexpectedly!")

threading.Thread(target=monitor_threads, daemon=True).start()
#%%


if __name__ == '__main__':
    
    # Start main threads
    udp_thread.start()
    detection_thread.start()
    record_thread.start()
    upload_thread.start()
    
    # Start monitoring thread health
    threading.Thread(target=monitor_threads, daemon=True).start()

    # Setup graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to stop.")
    signal.pause()
