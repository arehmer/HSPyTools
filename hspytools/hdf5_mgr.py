# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:11:32 2023

@author: Rehmer
"""
import pandas as pd
from pathlib import Path
import h5py
import numpy as np
import warnings
import os

from hspytools.readers import HTPAdGUI_FileReader

class hdf5_mgr():

    def __init__(self,hdf5_path,**kwargs):
        """
        Initializes the container and links it to the hdf5 file specified via 
        the given path.
        
        Parameters
        ----------
        path : str
            path to the hdf5-file that already exists or will be created by
            calling this method.
        mode : char
            A character specifying what should be done if the file already
            exists. 
            'a': Read/write if exists, create otherwise
            'w': overwrite
            
        Returns
        -------
        None.
        """
        
        mode = kwargs.pop('mode','a')
        width = kwargs.pop('width',None)
        height = kwargs.pop('height',None)
        
        if width == None or height == None:
            self.tparray = None
        else:
            self.tparray = TPArray(width,height)
                
        # Check if file exists
        if os.path.isfile(hdf5_path) == True:
            file_exists = True
            print('Specified file already exists.' )
        else:
            file_exists = False
            
        if file_exists and mode == 'w':
            print('Existing file is deleted.')
            os.remove(hdf5_path)
            
        
        # open hdf5_file or create if it doesn't exist
        hdf5_file = h5py.File(hdf5_path, "a")
        
        # Check if hdf5_file is empty
        if len(hdf5_file.keys()) == 0:
            is_empty = True
        else:
            is_empty = False
            # Close hdf5 file
        
        # Save path of hdf5 file for future reference
        self._hdf5_path = hdf5_file.filename
        
        hdf5_file.close()
        
        if is_empty:
            self._initialize_index()
            self._initialize_types()
    
    # @property
    # def types(self):
    #     return self._read_dict_from_hdf5('types')
    
    # @types.setter
    # def types(self,types):
    #     # Load current type dict from file
    #     types_old = self._read_dict_from_hdf5('types')
        
    #     # Update old dict (in-place operation)
    #     types_old.update(types)
        
    #     # Write new dict back do dictionary        
    #     self._save_attributes_to_hdf5('types',types_old)
    
            

    def _write_dict_to_hdf5(self,target_group,dictionary):
        """
        Writes the contents of a dictionary to a group called 'types' in an HDF5 file.
        
        Args:
            
        """
        
        with h5py.File(self._hdf5_path, 'w') as hdf_file:
            types_group = hdf_file.create_group(target_group)
            for key, value in dictionary.items():
                types_group.attrs[key] = value
    
    def _read_dict_from_hdf5(self,target_group):
        """
        Reads the contents of the 'types' group in an HDF5 file into a dictionary.
        
        Args:
            filename (str): Name of the HDF5 file to read from.
        
        Returns:
            dict: Dictionary with string keys and string values.
        """
        dictionary = {}
        with h5py.File(self._hdf5_path, 'r') as hdf_file:
            types_group = hdf_file[target_group]
            for key, value in types_group.attrs.items():
                dictionary[key] = value
                
        return dictionary
    
    
    def _initialize_index(self):

        df_index = pd.DataFrame(data = [])
        df_index.to_hdf(self._hdf5_path, key = 'index')

    def _initialize_types(self):
        """
        Initializes an empty group in the hdf5 file

        Returns
        -------
        None.

        """
        self._write_dict_to_hdf5('types',{})
    
    def _write_fields(self,field_dict):
        """
        Writes pandas dictionary arranged in field_dict to hdf5. Key of 
        field_dict is address in hdf5 to write to. Value is Data Frame to write
        """
        
        # Since a string is written to a dataframe and then saved in a hdf5-file
        # this will raise a PerformanceWarning, which is lengthy and should not
        # be displayed in the console
        
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")
            
            for address in field_dict.keys():
                field_dict[address].to_hdf(self._hdf5_path,address)
                
    def load_df(self,address,**kwargs):
        
        return pd.read_hdf(self._hdf5_path, address)
    
    def load_index(self):
        
        address = 'index'
        
        df_index = pd.read_hdf(self._hdf5_path, address)
        
        if len(self.hdf5Index_dtypes)!=0 and len(df_index.columns)!=0:
            df_index = df_index.astype(self.hdf5Index_dtypes)
       
        return df_index
    
    def load_meas(self,idx,**kwargs):
    
        # Get adresses
        df_index = self.load_index()
        address = df_index.loc[idx,'address']
        
        # appendix = kwargs.pop('appendix','')
        
        # address = 'meas/' + meas_name
    
        # Load and return specified video sequence
        return pd.read_hdf(self._hdf5_path, address + '/data')
    
    def load_LuT(self,lut_name):
        
        lut_address = '/LuT/' + lut_name
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            keys = list(hdf5_file[lut_address].keys())
        
        # Load dfs for every key into dictionary 
        LuT = {key:self.load_df(lut_address + '/' + key) for key in keys}
        
        return LuT
    
    def load_BCC(self,bcc_name):
        
        bcc_address = '/BCC/' + bcc_name
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            keys = list(hdf5_file[bcc_address].keys())
        
        # Load dfs for every key into dictionary 
        BCC = {key:self.load_df(bcc_address + '/' + key) for key in keys}
        
        return BCC
    
    def delete_BCC(self,bcc_name):
        
        # Complete name to full hdf5_address
        bcc_address = '/BCC/' + bcc_name
        
        # load bds index
        bds_index = self.load_index()
        
        # Delete the BCC in the hdf5 file and references to it in bds_index 
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            try:
                del hdf5_file[bcc_address]
                bds_index.loc[bds_index['BCC'] == bcc_name,'BCC'] = None
            except:
                print(bcc_name + " couldn't be deleted or doesn't exist anymore")
                bds_index.loc[bds_index['BCC'] == bcc_name,'BCC'] = None
        
        # write the new bds index to the file
        self._write_fields({'index':bds_index})
        
        return None
    
    def _create_group_by_copy(self,source,target,groups,**kwargs):
                
        
        with h5py.File(self._hdf5_path,  'a') as hdf5_file:
                        
            # copy specified groups in source to target
            for group in groups:
                hdf5_file.copy(source+'/'+group,target+'/'+group)
            
        return None
    
        
    def _group_exists(self,hdf5_address:str)->bool:
        """
        Checks if a group already exists at the given address in the hdf5-file
    
        Parameters
        ----------
        hdf5_address : str
            DESCRIPTION.
    
        Returns
        -------
        bool
            DESCRIPTION.
    
        """
        
        group_exists = False
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            if hdf5_address in hdf5_file:
                group_exists = True
        
        return group_exists
    
    def video_to_avi(self,unique_id,**kwargs):
        """
        Writes a video sequence already stored in the hdf5-file under the name
        video_name to an mp4-file. Video will be saved in the folder specified 
        in folder_path under the file name <video_name>.mp4
        
        Parameters
        ----------
        video_name : str
            name under which the video is stored in the hdf5-file. If a 
            postprocessed version of the video is to be addressed, then pass
            'video_name/filtered' for example, or 'video_name/gradient'
        folder_path : str
            Path to the folder where the video should be saved. 
    
            
        Returns
        -------
        None.
        """
        
        print('''For annotation purposes it is highly recommended to export the 
              video sequence as png-images via write_to_png(). mp4 performs
              interpolations between frames that to not reflect the original
              sensor image and can deaviate from it.''')
        
        fs = kwargs.pop('fs',8)
        appendix = kwargs.pop('appendix','')
        
        
        # appendix = kwargs.pop('appendix','')
        
        # Convert device to list
        if not isinstance(unique_id,list):
            unique_id = [unique_id]
        
        # Load index of hdf5 file
        index = self.load_index()
        
        for u_id in unique_id:
            
            # Get address
            a = index.loc[u_id,'address']
            
            # Load specified video sequence
            df_video = self.load_df(a+'/data')
            
            # Get size information
            (w,h) = (self.tparray.width,self.tparray.height)
            
            # Initialize a HTPAdGUI_FileReader, which has a function for 
            # writing DataFrames of video to .png
            reader = HTPAdGUI_FileReader(w,h)
            
            # Pass the DataFrame to the method for writing to .png
            path = Path.cwd() /  (a+'/avi')
            video_name = a.split('/')[-1]
            reader.export_avi(df_video,video_name,path)
            
        return None
        
    
    
    def video_to_png(self,unique_id,**kwargs):
        """
        Writes a video sequence already stored in the hdf5-file under the name
        video_name to png-images frame by frame. The files will be saved 
        in a folder with the name <video_name> in the folder specified in
        folder_path. The png-files will be named according to their frame 
        number.
        
        Parameters
        ----------
        video_name : str
            name under which the video is stored in the hdf5-file. If a 
            postprocessed version of the video is to be addressed, then pass
            'video_name/filtered' for example, or 'video_name/gradient'
        folder_path : str
            Path to the folder where the video should be saved. 
    
            
        Returns
        -------
        None.
        """
        
       
        # Load index of hdf5 file
        index = self.load_index()
        
        for u_id in unique_id:
            
               
            # Get address
            a = index.loc[u_id,'address']
            
            # Load specified video sequence
            df_video = self.load_df(a+'/data')
            
            # Get size information
            (w,h) = (index.loc[u_id,'Width'],index.loc[u_id,'Height'])
            
            # Initialize a HTPAdGUI_FileReader, which has a function for 
            # writing DataFrames of video to .png
            reader = HTPAdGUI_FileReader(w,h)
            
            # Pass the DataFrame to the method for writing to .png
            path = Path.cwd() /  (a+'/png')
            reader.export_png(df_video,
                              path,
                              **kwargs)
            
        return path