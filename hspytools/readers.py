# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:08:22 2024

@author: rehmer
"""


import numpy as np 
from pathlib import Path
import pandas as pd
import os
import matplotlib
import socket
import re

from .tparray import TPArray

class HTPAdGUI_FileReader():
    
    def __init__(self,width,height):
        
        self.width = width
        self.height = height
        
        
        # Depending on the sensor size, the content of the bds-file is
        # organized as follows
        self.tparray = TPArray(width,height)
        # data_order = ArrayType.get_serial_data_order()
                
    def read_htpa_video(self,path):
        
        # Check if path is provided as pathlib
        if not isinstance(path,Path):
            path = Path(path)
        
        # get suffix
        suffix = path.suffix

        if suffix.casefold() == ('.txt').casefold():
            df_video, header = self._import_txt(path) 
        elif suffix.casefold() == ('.bds').casefold():
            df_video, header = self._import_bds(path)
        else:
            print('File extension not recognized.')
            return None
        
        return df_video, header
    
    def _import_txt(self,path,**kwargs):
        
        sep = kwargs.pop('sep',' ')
        skiprows = kwargs.pop('skiprows',1)
        
        # Get columns names
        columns = self.tparray.get_serial_data_order()
        
        # Read txt with pandas to DataFrame
        txt_content = pd.read_csv(path,
                                  sep=sep,
                                  skiprows=skiprows,
                                  header=None)
        
        # txt file contains a time stamp at the end, that's not in the bds-file
        # use only the first columns that contain actual data
        txt_content = txt_content[np.arange(0,len(columns),1)]
        
        # rename columns appropriately
        txt_content.columns = columns

        # rename index appropriately
        txt_content.index = range(0,len(txt_content))
        txt_content.index.name = 'image_id'
        
        header = None
        
        return txt_content, header
    
    def _import_bds(self,bds_path,**kwargs):
        
        # open file and save content byte by byte in list
        bds_content = []
        
        with open(bds_path, "rb") as f:
            
            # Read the header byte by byte until '\n'
            header = []
            header_end = False
            
            while header_end == False:
                header.append(f.read(1))

                if header[-1].decode()=='\n':
                    header_end = True
            
            # Join bytes of header together
            header = bytes().join(header)

                
            # Read two bytes at a time 
            while (LSb := f.read(2)):
                
                # and combine in LSb fashion
                bds_content.append(int.from_bytes(LSb, 
                                                  byteorder='little'))
               
        # Cast the data to a DataFrame of appropriate size
        columns = self.tparray.get_serial_data_order()
        
        # If the last frame has not been fully transmitted, reshaping fails
        # Therefore throw out the last incomplete frame
        num_full_frames = int(len(bds_content)/len(columns))
        bds_content = bds_content[0:num_full_frames*len(columns)]
        bds_content = (np.array(bds_content)).reshape(-1,len(columns))
        bds_content = pd.DataFrame(data=bds_content,
                                   columns = columns)
        
        bds_content.index = range(len(bds_content))
        bds_content.index.name = 'image_id'
        
        return bds_content, header
        
    def reverse(self,df_video):
        """
        Function for rotating a video by 180°. Intended for postprocessing 
        a video sequence from a sensor that was mounted upside-down

        Parameters
        ----------
        df_video : pd.DataFrame
            DataFrame containing the content of a text or bds file exported 
            from the HTPAdGUI

        Returns
        -------
        df_video: pd.DataFrame
            Upside-down version of the original video.

        """
        
        # Rotation must be applied to the pixel values and the electrical offsets
        pix_cols = self.tparray._pix
        e_off_cols = self.tparray._e_off
        
        NROFBLOCKS = self.tparray._DevConst['NROFBLOCKS']
        width = self.tparray._width
        height = self.tparray._height
        
        # go through the DataFrame image by image
        for i in df_video.index:
            
            # get pixel and electrical offsets values
            pix_val = df_video.loc[i,pix_cols]
            e_off_val = df_video.loc[i,e_off_cols]
            
            # reshape values according to sensor
            pix_val = pix_val.values.reshape(self.tparray._npsize)
            e_off_val = e_off_val.values.reshape((int(height/NROFBLOCKS),
                                                  width))
            
            # rotate both arrays by 180°
            pix_val = np.rot90(pix_val,k=2)
            e_off_val = np.rot90(e_off_val,k=2)
            
            # reshape to a row vector and write to dataframe
            df_video.loc[i,pix_cols] = pix_val.flatten()
            df_video.loc[i,e_off_cols] = e_off_val.flatten()
            
        # Return the rotated dataframe
        return df_video
    
    def export_bds(self,df_video,header,bds_path,**kwargs):
        """
        Export a video sequence in dataframe format to a .bds file that is
        compatible with HTPAdGUI
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.
        header : byte
            DESCRIPTION.
        path : pathlib.Path
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        mode = kwargs.pop('mode','x')
        
        
        # Check if file already exists
        if bds_path.exists():
            if mode=='x':
                print('File exists. No data will be written. Pass mode="x" to overwrite.')
            elif mode=='w':
                print('File exists and will be overwritten.')
                os.remove(bds_path)
        
        # first write the header to the list byte by byte
        bds_content = []
        bds_content.append(header)
        # Go over the video sequence image by image, convert all integers to 
        # bytes and append to the list
        
        for i in df_video.index:
            
            # get the whole row as a row vector
            row = df_video.loc[i].values
            
            # cast every integer to a byte in little endian byteorder
            for val in row:
                bds_content.append(int(val).to_bytes(length=2,byteorder='little'))
            
        # Write bytes to file
        with open(bds_path, "wb") as bds_file:
            [bds_file.write(b) for b in bds_content]
            
        return None
    
    def export_png(self,df_video,path):
        """
        A function for writing a video sequence given as a DataFrame to .png
        frame by frame in a sepcified folder (path)
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Get shape of sensor array
        npsize = self.tparray._npsize
        
        # Get columns with pixel values
        pix_cols = self.tparray._pix
        
        # Get rid of everything else
        df_video = df_video[pix_cols]
        
       
       
       
        if not path.exists():
            path.mkdir(parents=True,exist_ok=False)
        
        for i in df_video.index:
            img = df_video.loc[[i]].values.reshape(npsize)

            
            file_name = str(i) + '.png'
            
            img = self._scale_img(img)

            matplotlib.image.imsave(path / file_name, img)
        
        return None

    # def export_avi(self,df_video,video_name,path,**kwargs):
    #     """
    #     A function for writing a video sequence given as a DataFrame to .avi
    #     in a sepcified folder (path)
        

    #     Parameters
    #     ----------
    #     df_video : pd.DataFrame
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """        

    #     # Framerate
    #     fps = kwargs.pop('fps',8)
        
    #     # Get shape of sensor array
    #     size = self.tparray._size
    #     npsize = self.tparray._npsize
        
    #     # Get columns with pixel values
    #     pix_cols = self.tparray._pix
        
    #     # Get rid of everything else
    #     df_video = df_video[pix_cols]
        
    #     if not path.exists():
    #         path.mkdir(parents=True,exist_ok=False)
        
    #     # Initialize video writer
    #     writer = imageio.get_writer((path / (video_name + '.avi')).as_posix(),
    #                                 fps=fps,
    #                                 macro_block_size=1)
        
    #     # codec = cv2.VideoWriter_fourcc(*'MJPG')
    #     # codec = cv2.VideoWriter_fourcc(*'H264')
        
    #     # video = cv2.VideoWriter((path / (video_name + '.avi')).as_posix(),
    #     #                         codec,
    #     #                         fs,
    #     #                         size,
    #     #                         isColor = True)  
                
    #     # Get colormap
    #     cmap = matplotlib.cm.get_cmap('plasma')      
        
    #     # Loop over all frames and write to mp4
    #     for i in df_video.index:
            
    #         img = df_video.loc[i].values.reshape(npsize)
            
    #         # Normalization 
    #         img = ( img - img.min() ) / (img.max() - img.min())
            
    #         # Apply colormap
    #         RGBA = (cmap(img)*255).astype('uint8')
            
    #         # opencv
    #         # BGR = cv2.cvtColor(RGBA, cv2.COLOR_RGB2BGR)
    #         # video.write(BGR)
            
    #         # imageio_ffmpeg
    #         writer.append_data(RGBA)

    #     # opencv
    #     # cv2.destroyAllWindows()
    #     # video.release()
        
    #     # imageio_ffmpeg
    #     writer.close()
       
    #     return None   
    
    def _scale_img(self,img):
        
        # Get width an height of image
        w = self.tparray._width
        h = self.tparray._height
        
        # Crop image by 10 % circumferential
        crop_w = int(np.ceil(0.1*w))
        crop_h = int(np.ceil(0.1*h))
        
        # Crop the image by 3 pixels to get rid of corners.
        img_crop = img[crop_h:h-crop_h,crop_w:w-crop_w]
        
        # Use the pixel values in the cropped frame to scale the image
        dK_max  = img_crop.max()
        dK_min  = img_crop.min()
        
        img = ( img - dK_min ) / (dK_max - dK_min)
        
        img[img<=0] = 0


        return img
    
    def _flip(self,df_video):

        w = self.width
        h = self.height
        
        pix_cols = self.tparray._pix
        
        for i in df_video.index:
            img = df_video.loc[i,pix_cols].values.reshape((h,w))
            img = np.flip(img,axis=1).flatten()
            df_video.loc[i,pix_cols] = img

            
        return df_video
               
 
class HTPA_ByteStream_Converter():
    def __init__(self,width,height,**kwargs):
        
        self.width = width
        self.height = height
        
        self.tparray = TPArray(width,height)
        
        # Initialize an array to which to write data
        self.data_cols = self.tparray.get_serial_data_order()
        self.data = np.zeros((len(self.data_cols)))
        
        self.output = kwargs.pop('output','np')
        
        # Depending on the specified output format, the convert method is 
        # pointing to _bytes_to_np() or _bytes_to_pd()
        if self.output == 'np':
            self.convert = self._bytes_to_np
        elif self.output == 'pd':
            self.convert = self._bytes_to_pd
            
    
    def _bytes_to_np(self,byte_stream:list):
        
        if not isinstance(byte_stream, list):
            TypeError('byte_stream needs to be a list of bytes')
                
        # Zero data array
        self.data.fill(0)
        
        j=0
        
        # Loop over all elements / packages in list
        for package in byte_stream:
            
            # Read the first byte, it's the package index, and throw away
            _ = package[0]
            
            # Loop over all bytes and combine MSB and LSB
            idx = np.arange(1,len(package),2)
            
            for i in idx:    
                self.data[j] = int.from_bytes(package[i:i+2], byteorder='little')
                j = j+1
       
        Warning("Check if pixels are in appropriate order (i.e. if Bodo's)"+\
                "and pyplots/opencvs coordinate system are the same")
        
        return self.data
    
    def _bytes_to_pd(self,byte_stream:list):
        
        self._bytes_to_np(byte_stream)
        
        df_data = pd.DataFrame(data=[self.data],
                               columns=self.data_cols)
        
        return df_data
        
        
    # def bytes_to_img(self,byte_stream):
        
             
    #     # Loop over all bytes and combine MSB and LSB
    #     idx = np.arange(0,len(byte_stream),2)
        
    #     img = np.zeros((1,self.width*self.height))
    #     j=0
        
    #     for i in idx:    
    #         img[0,j] = int.from_bytes(byte_stream[i:i+2], byteorder='little')
    #         j = j+1
            
    #     img = img.reshape((self.height,self.width))
    #     img = np.flip(img,axis=1)
        
    #     return img
    
    
class HTPA_UDPReader():
    
    def __init__(self,width,height,**kwargs):
        
        self.width = width
        self.height = height
               
        # Port over which HTPA device connects, usually 30444
        self._port = kwargs.pop('port',30444)
        
        # Initialize a TPArray object, which contains all information regarding
        # how data is stored, organized and transmitted for this array type
        self._tparray = TPArray(width,height)
        
        # Create a DataFrame to store all bound devices in
        self.col_dict = {'IP':str,'MAC-ID':str,'Arraytype':int,
                         'status':str}
        self.index = pd.Index(data=[],dtype=int,name='DevID')
        self._devices = pd.DataFrame(data=[],
                                    columns = self.col_dict.keys(),
                                    index = self.index)
        
        # Dictionary for storing all sockets in
        self._sockets = {}
    
        # Initialize ByteStream Reader for convertings bytes to pandas
        # dataframes
        self.bytestream_converter = \
            HTPA_ByteStream_Converter(width,height,**kwargs)
        
        # depending on the desired output type, choose which method of
        # HTPA_ByteStream_Reader should be used to parse bytes to that 
        # type
        
        # if self.output == 'np':
        #     self.bytes_to_output = self.bytestream_reader.bytes_to_np
        # if self.output == 'pd':
        #     self.bytes_to_output = self.bytestream_reader.bytes_to_pd
            
        
    @property
    def output(self):
        return self._output    
    @property
    def port(self):
        return self._port
    @property
    def tparray(self):
        return self._tparray
    
    
    @property
    def devices(self):
        return self._devices
    @devices.setter
    def devices(self,df):
        devices_old = self._devices
        self._devices = pd.concat(devices_old,df)

    @property
    def sockets(self):
        return self._sockets
    @sockets.setter
    def sockets(self,socket:dict):
        self._sockets.update(socket)

    
    def broadcast(self):
        Warning('Not implemented yet.')
        return None
    
    def _clear_port(self,udp_socket):
        """
        Read all packages available at the specified port. 

        Parameters
        ----------
        server_address : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        clear = False
        while clear == False:
            
            try:
                package = udp_socket.recv(self.tparray._package_size)
            except:
                clear = True
                
        return package
            
    
    def bind_tparray(self,ip:str):
        """
        Creates a socket for the device with the given ip

        Parameters
        ----------
        ip : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Create the udp socket
        udp_socket = socket.socket(socket.AF_INET,          # Internet
                                   socket.SOCK_DGRAM)       # UDP
        
        # Create server address
        server_address = (ip,self._port)
               
        # Set timeout to 1 second
        # socket.setdefaulttimeout(1)
        udp_socket.settimeout(1)
        
        # Try calling the device and check if it's a HTPA device
        try:
            _ = udp_socket.sendto(bytes('Calling HTPA series devices',
                                        'utf-8'),
                                       server_address)
        except:
            Exception('Calling HTPA series device failed')
            return None
        
        # Read in all packages at this server address to clear the port. 
        # Keep only the last package which should be the answer to the call
        call = self._clear_port(udp_socket)
        
        
        # If calling was successfull, extract basic information from the 
        # answer string
        dev_info = self._callstring_to_information(call)
        
        # Next try to bind the device
        try:
            _ = udp_socket.sendto(bytes('Bind HTPA series device',
                                        'utf-8'),
                                       server_address)
            bind = udp_socket.recv(self.tparray._package_size)
        except:
            Exception('Calling HTPA series device failed')
            return None
        
        dev_info['status'] = 'bound'
        
        # Add socket to dictionary
        dev_id = dev_info.index.item()
        self.sockets = {dev_id:udp_socket}
        
        # Append new device to device list and store
        self._devices = dev_info
        
        return self.devices.copy()

    def release_tparray(self,dev_id):
        
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
            
            # Get udp socket
            udp_socket = self.sockets[dev_id]
            
            # Create server address
            server_address = (dev_info['IP'].item(), self.port)
            
            # Send message to device to stop streaming bytes
            _ = udp_socket.sendto(bytes('x','utf-8'),server_address)
    
    def start_continuous_bytestream(self,dev_id):
        
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
                
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('K','utf-8'),server_address)

    def stop_continuous_bytestream(self,dev_id):
        
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
                
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)
                
        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('X','utf-8'),server_address)        
        
        return None
    
    def read_continuous_bytestream(self,dev_id):
        
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
        
        # Create variable that indicates if DataFrame was constructed success-
        # fully
        success = False
        while success == False:
            
            # Initialize list for received packages            
            packages = []
            
            # Read incoming packages until one with the package index 1 is received
            sync = False
            
            while sync == False:
                
                # Receive package
                package = udp_socket.recv(self.tparray._package_size)
                
                # Get index of package
                package_index = int(package[0])
                
                # Check if it is equal to 1
                if package_index == 1:
                    packages.append(package)
                    sync = True
            
            for p in range(2,self.tparray._package_num+1):
                
                # Receive package
                package = udp_socket.recv(self.tparray._package_size)
                
                # Get index of package
                package_index = int(package[0])
                
                # Check if package index has the expected value
                if package_index == p:
                    packages.append(package)
                
            # In the end check if as many packages were received as expected
            if len(packages) == self.tparray._package_num:
                
                # If yes, pass the packages to the class that parses the
                # bytes into an np.ndarray or pd.DataFrame
                # df_frame = self.bytestream_reader.bytes_to_df(packages)
                
                frame = self.bytestream_converter.convert(packages)
                
                success = True
                
            else:
                print('Frame lost.')
                sync = False


        return frame
        
    def read_single_frame(self,dev_id,**kwargs):
        
        # Read voltage 'c' or temperature 'k' frame
        mode = kwargs.pop('mode','k')
        
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
        
        # Create server address, i.e. tuple of device IP and port 
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to send the single frame
        mess = udp_socket.sendto(bytes(mode,'utf-8'),server_address)        
        
        # Michaels code is atrocious. Sometimes it sends no frame, sometimes
        # 1 frame and sometimes two frames. All these cases have to be considered
        # Try to get first frame (works always)
        try:
            frame = self._receive_udp_frame(udp_socket)
        except:
            frame = pd.DataFrame(data=[],
                                 columns = self.tparray.get_serial_data_order())
        
        if frame is None:
            frame = pd.DataFrame(data=[],
                                 columns = self.tparray.get_serial_data_order())
        
        if len(frame) == 0:
            print('Reading frame failed.')
        else:
            print('Successfully received frame')
        
        # Clean up whole port in which a second frame may or may not be
        self._clear_port(udp_socket)
        
        return frame
    
    def _receive_udp_frame(self,udp_socket):
        """
        Receives UDP packages and tries to put them together to a frame

        Parameters
        ----------
        udp_socket : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Initialize list for received packages            
        packages = []
        
        # Read incoming packages until one with the package index 1 is received
        sync = False
        
        while sync == False:
            
            # Receive package
            package = udp_socket.recv(self.tparray._package_size)
            
            # Get index of package
            package_index = int(package[0])
            
            # Check if it is equal to 1
            if package_index == 1:
                packages.append(package)
                sync = True
        
        for p in range(2,self.tparray._package_num+1):
            
            # Receive package
            package = udp_socket.recv(self.tparray._package_size)
            
            # Get index of package
            package_index = int(package[0])
            
            # Check if package index has the expected value
            if package_index == p:
                packages.append(package)
                
        # In the end check if as many packages were received as expected
        if len(packages) == self.tparray._package_num:
                            
            frame = self.bytestream_converter.convert(packages)
                        
        else:
            print('Frame lost.')
            frame = None


        return frame
        
    
    def device_put_to_sleep(self,dev_id,**kwargs):
        """
        Puts a device to sleep
        """
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
                
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('s','utf-8'),server_address)
        answer = udp_socket.recv(self.tparray._package_size)
        answer = answer.decode('utf-8')
        
        
        if answer.strip() == 'Module set into sleep mode.':
            print('Device ' + str(dev_id) + ' put to sleep.')
            return True
        else:
            print('Putting device ' + str(dev_id) + ' to sleep failed.')
            return False
        
    def device_wake_up(self,dev_id,**kwargs):
        """
        Wakes device up
        """
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
                
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('S','utf-8'),server_address)
        answer = udp_socket.recv(self.tparray._package_size)
        answer = answer.decode('utf-8')
        
        
        if answer.strip() == 'Module woke up.':
            print('Device ' + str(dev_id) + ' awake.')
            return True
        else:
            print('Failed waking up device ' + str(dev_id) + '.')
            return False
    
    
    def _callstring_to_information(self,call:bytes):
        """
        Extracts information from the ridiculously long and inefficient answer
        to the device call.

        Parameters
        ----------
        call : bytes
            DESCRIPTION.

        Returns
        -------
        dev_info : TYPE
            DESCRIPTION.

        """

        call = call.decode('utf-8')
        
        # Extract information from callstring
        arraytype = int(call.split('Arraytype')[1].split('MODTYPE')[0])
        mac_id = re.findall(r'\w{2}\.\w{2}\.\w{2}\.\w{2}.\w{2}.\w{2}',call)[0]
        ip = re.findall(r'\d{3}\.\d{3}\.\d{3}.\d{3}',call)[0]
        
        # Remove leading zeros from IP
        ip = '.'.join([str(int(x)) for x in ip.split('.')])
        
        
        dev_id = int(call.split('DevID:')[1].split('Emission')[0])
        
                
        dev_info = pd.DataFrame(data = [],
                                columns = self.col_dict.keys(),
                                index = self.index)
        
        dev_info.loc[dev_id,['IP','MAC-ID','Arraytype']] = \
            [ip,mac_id,arraytype]
        
        dev_info = dev_info.astype(self.col_dict)    
        
        return dev_info