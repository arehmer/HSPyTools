# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:57:21 2025

@author: rehmer
"""

from queue import Queue
from threading import Condition

import time

from hspytools.readers import HTPA_UDPReader
from hspytools.ipc.threads import UDP, Imshow
from hspytools.tparray import TPArray

# %% Parameters provided by the user
# Resolution of sensor
width = 60
height = 40

# Broadcasting has not been implemented yet, therefore the IP of the HTPA
# device needs to be known in advance

# %% Scan for devices
tparray = TPArray(width = width,
                  height = height)

udp_buffer = Queue(maxsize=1)

# Create instance of UDP reader
udp_reader = HTPA_UDPReader(width,height)

# Scan for htpa devices in the subnet 192.168.240.0/24
udp_reader.broadcast('192.168.240.255')
# print(udp_reader.devices)

# Assuming a device has been found, get its DevID and IP
try:
    DevID = udp_reader.devices.iloc[0].name
    IP = udp_reader.devices.iloc[0]['IP']
    
    print('Found a device with DevID ' + str(DevID) + \
          ' and IP ' + str(IP))
    
except:
    raise Exception('No device has been found in this subnet!')


#%% Create a buffer that the udp thread writes into and the plot thread reads
# from
udp_buffer = Queue(1)
udp_lock = Condition()

# Create instance of UDP thread
udp_thread = UDP(udp_reader = udp_reader,
                  IP = IP,
                  write_buffer = udp_buffer,
                  write_condition = udp_lock)


# Create instance of thread plotting the reveived data
plot_thread = Imshow(width = width,
                      height = height,
                      read_buffer = udp_buffer,
                      read_condition = udp_lock)

if __name__ == '__main__':
    
      
    udp_thread.start()
    plot_thread.start()
    
    for i in range(1):
        time.sleep(20)
        udp_thread.stop()
        plot_thread.stop()