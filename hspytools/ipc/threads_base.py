# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:12:35 2024

@author: rehmer
"""

from threading import Thread
from queue import Queue


class RWThread(Thread):
    """
    Base class for a thread that reads from one buffer and writes into another
    """
    def __init__(self,target,
                 read_buffer:Queue,
                 write_buffer:Queue,
                 **kwargs):

        self.read_buffer = read_buffer
        self.write_buffer = write_buffer
        self._exit = False               # Exit flag 

        super().__init__(target=target,**kwargs)
    
    def run(self):
        
        while self._exit == False:
            
            # Execute target function
            result = self._target()
            
            # Write result to buffer
            self.write_buffer.put(result)
            
            # Signal that processing on this item in the read_buffer is done
            self.read_buffer.task_done()
            
    def _target_function(self):
        
        print('You need to overwrite this method in the child class.')
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict
                        
    def stop(self):
        self._exit = True
        
  

class WThread(Thread):
    """
    Base class for a thread that writes into a buffer
    """
    def __init__(self,target,write_buffer:Queue,**kwargs):
        
        self.write_buffer = write_buffer
        self._exit = False 
        
        super().__init__(target=target,**kwargs)
        
    def run(self):
        
        while self._exit == False:
                        
            # Execute target function
            result = self._target()
            
            # Write result to buffer
            self.write_buffer.put(result)
                       
    def _target_function(self):
        # This is a basic target function, that only gets the data from a buffer
        # an passes it on. If any further processing is required, this target
        # function or the run method must be overwritten in a child class
        
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict
    
    
    def stop(self):
        self._exit = True
        
        
class RThread(Thread):
    """
    Base class for a thread that reads from a buffer
    """
    def __init__(self,target,read_buffer:Queue,**kwargs):
        
        self.read_buffer = read_buffer
        self._exit = False 
        
        super().__init__(target=target,**kwargs)
        
    def run(self):
        
        while self._exit == False:
                        
            # Execute target function
            result = self._target()
            
            # Signal that processing on this item in the read_buffer is done
            self.read_buffer.task_done()
                       
    def _target_function(self):
        
        print('You need to overwrite this method in the child class.')
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict
            
    def stop(self):
        self._exit = True