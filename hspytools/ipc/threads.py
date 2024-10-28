# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:13:39 2024

@author: rehmer
"""
import time
import cv2
from queue import Queue
import numpy as np
import pandas as pd

from hspytools.ipc.threads_base import WThread,RThread
from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray



class UDP(WThread):
    """
    Class for running HTPA_UDP_Reader in a thread.
    """
    
    def __init__(self,udp_reader:HTPA_UDPReader,
                 dev_ip:str,
                 write_buffer:Queue,
                 **kwargs):
        """
        

        Parameters
        ----------
        udp_reader : HTPA_UDP_Reader
            DESCRIPTION.
        dev_ip : str
            DESCRIPTION.
        start_trigger : Event
            DESCRIPTION.
        finish_trigger : Event
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # self.output_type = kwargs.pop('output_type','np')
        
        # Set UDP Reader object as attribute
        self.udp_reader = udp_reader
        
        # Use object to bind htpa device available under dev_ip
        bound_devices = self.udp_reader.bind_tparray(dev_ip)
        self.dev_id = bound_devices[bound_devices['IP']==dev_ip].index.item()
        
        
        # Start continuous bytestream 
        self.udp_reader.start_continuous_bytestream(self.dev_id)
               
        
        # Set time
        self.t0 = time.time()
        
        super().__init__(target = self._target_function,
                         write_buffer = write_buffer,
                         **kwargs)
            
    def _target_function(self):
        
        # print('Executed upd thread: ' + str(time.time()-self.t0) )
        
        frame = self.udp_reader.read_continuous_bytestream(self.dev_id)
        
        return {'frame':frame}
        
    def stop(self):
        
        # Set attribute exit to stop run method of thread
        self._exit = True
        
        # Stop the stream
        self.udp_reader.stop_continuous_bytestream(DevID = self.dev_id)
        
        # Release the array
        self.udp_reader.release_tparray(self.dev_id)
        
        
class Imshow(RThread):
    """
    Class for running HTPA_UDP_Reader in a thread.
    """
    
    def __init__(self,
                 width:int,
                 height:int,
                 read_buffer:Queue,
                 **kwargs):
        
        self.tparray = TPArray(width = width, height = height)
        self.num_pix = len(self.tparray._pix)
        
        self.window_name = kwargs.pop('window_name','Sensor stream')
        
        # Call parent class
        super().__init__(target = self._target_function,
                         read_buffer = read_buffer,
                         **kwargs)
    
    def _target_function(self):
                
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
        frame = upstream_dict['frame']

        # Reshape if not the proper size
        if frame.ndim == 1:
            frame = frame[0:self.num_pix]
            frame = frame.reshape(self.tparray._npsize)
             
        # convert to opencv type
        frame = cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)
        
        # Get bounding boxes
        try:
            bboxes = upstream_dict['bboxes']
        except:
            bboxes = pd.DataFrame(data=[])
        
        # print('Executed plot thread \n')
        
        return frame,bboxes
    
    def run(self):
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while self._exit == False:
            
            # Execute target function
            frame,bboxes = self._target()
            
            # Add a rectangle for every box
            for b in bboxes.index:
                
                box = bboxes.loc[[b]]
            
                x,y = box['xtl'].item(),box['ytl'].item(),
                w = box['xbr'].item() - box['xtl'].item()
                h = box['ybr'].item() - box['ytl'].item()

                frame = cv2.rectangle(frame, (x,y), (x+w,y+h), 1 ,1)
            
            cv2.imshow(self.window_name,frame)
            cv2.waitKey(1)

        # The opencv window needs to be closed inside the run function,
        # otherwise a window with the same name can never be opened until
        # the console is restarted
        if self._exit == True:
            cv2.destroyWindow(self.window_name)
            
        def stop(self):
            self._exit = True


