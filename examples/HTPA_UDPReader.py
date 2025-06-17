# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""

from queue import Queue

from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray


"""
This script is intended to show the basic functionality of the HTPA_UDPReader
class, i.e. how to broadcast for htpa devices, bind an htpa device, and start/
stop the sensor stream.

For actual applications the HTPA_UDPReader needs to be wrapped in a thread. 
This is demonstrated in HTPA_Threads.py
"""


# %% Parameters provided by the user
# Resolution of sensor
width = 60
height = 40

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


# %% Next, the device must be bound. This can be achieved either by using
# its DevID or its IP 

# Binding via IP
udp_reader.bind_tparray(IP = IP)
udp_reader.release_tparray(DevID)

# Binding via DevID
udp_reader.bind_tparray(DevID = DevID)
udp_reader.release_tparray(DevID)


        

    
    