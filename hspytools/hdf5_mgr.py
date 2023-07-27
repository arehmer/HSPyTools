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


class hdf5_mgr():
    
    def import_LuT(self,lut_path,device,**kwargs):
        """
        Import a Look up table to the hdf5-file that can be used to calculate
        temperature-values from raw measurement data
        

        Parameters
        ----------
        lut_path : pathlib.Path()
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        mode = kwargs.pop('mode','a')
                
        # Check if meas_path is given as pathlib path
        if not isinstance(lut_path,Path):
            print('lut_path must be given as pathlib.Path object')
            return None
        
        # Convert device to list
        if not isinstance(device,list):
            device = [device]
            
        
        # # Check if meas_path leads to a directory or a file
        # if lut_path.is_file():
        #     files = [lut_path]
        # elif lut_path.is_dir():
        #     files = [file for file in lut_path.iterdir()]
        
        
        # # For now, import the data just as Christoph did, for compatibility
        # for file in files:

        # Name in group is the file name without extension
        lut_name = lut_path.stem
        
        # Check if LuT already exists
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            
            if 'LuT/' + lut_name in hdf5_file:
                
                print('LuT ' + lut_name + ' already exists.\n')
                
                if mode == 'w':
                    print('Data is deleted and rewritten.\n')
                    self.delete_group('LuT/'+lut_name)
                else:
                    print('No data is written to group. Pass mode="overwrite" \n to overwrite existing data.')
                    return None
    
        
        # Load LuT from file
        LuT = self.tparray.import_LuT(lut_path)
        
        
        # Dict for fields to write
        data_to_write = {}
        
        # Convert every key in dictionary to pd DataFrame
        for key in LuT:
            if isinstance(LuT[key],np.ndarray):
                df_key = pd.DataFrame(LuT[key])
            else:
                df_key = pd.DataFrame([LuT[key]],columns=[key])
            
            data_to_write['/LuT/' + lut_name + '/' + key] = \
                df_key
        
        # Load the index, add information to which devices LuT belongs
        df_index = self.load_index()
        for d in device:
            df_index.loc[df_index['device']==d,'LuT'] = lut_name
        
        data_to_write['index'] = df_index
        
        # Write to hdf5
        try:
            self._write_fields(data_to_write)
            print(lut_name + ' successfully imported.')
        except:
            print('Some error occured when importing ' + lut_name + '.' )  
                
    def import_BCC(self,bcc_path,device,**kwargs):
        """
        Imports data from bcc file to the hdf5-file that can be used to calculate
        temperature-values from raw measurement data
        

        Parameters
        ----------
        bcc_path : pathlib.Path()
            DESCRIPTION.
        device : int
            Sensor ID

        Returns
        -------
        None.

        """
        
        mode = kwargs.pop('mode','a')
        
        # Check if meas_path is given as pathlib path
        if not isinstance(bcc_path,Path):
            print('bcc_path must be given as pathlib.Path object')
            return None
                   
        file = bcc_path
            
        # Name in group is the file name without extension
        bcc_name = file.stem
        
        # Check if BCC already exists
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            
            if 'BCC/' + bcc_name in hdf5_file:
                
                print('BCC ' + bcc_name + ' already exists.\n')
                
                if mode == 'w':
                    print('Data is deleted and rewritten.\n')
                    self.delete_group('BCC/'+bcc_name)
                else:
                    print('No data is written to group. Pass mode="overwrite" \n to overwrite existing data.')
                    return None
        
        # Get properties from bcc file
        bcc = self.tparray.import_BCC(bcc_path)
        
        # Dict for fields to write
        data_to_write = {}
        
        # Convert every key in dictionary to pd DataFrame
        for key in bcc:
            if isinstance(bcc[key],np.ndarray):
                df_key = pd.DataFrame(bcc[key])
            else:
                df_key = pd.DataFrame([bcc[key]],columns=[key])
            
            data_to_write['/BCC/' + bcc_name + '/' + key] = \
                df_key
        
        # Load the index, add information to which devices BCC belongs
        df_index = self.load_index()
        df_index.loc[df_index['device']==device,'BCC'] = bcc_name
        
        data_to_write['index'] = df_index
        # Write to hdf5
        try:
            self._write_fields(data_to_write)
            print(bcc_name + ' successfully imported.')
        except:
            print('Some error occured when importing ' + bcc_name + '.' )  
    
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
        
        return pd.read_hdf(self._hdf5_path, address)
    
    def load_meas(self,meas_name,**kwargs):

        appendix = kwargs.pop('appendix','')
        
        address = 'meas/' + meas_name

        # Load and return specified video sequence
        return pd.read_hdf(self._hdf5_path, address + '/data' + appendix)
    
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
    
    def delete_group(self,address):
        
        # delete from file
        with h5py.File(self._hdf5_path, 'a') as hdf5_file:
            del hdf5_file[address]
        
        # delete from index
        df_index = self.load_index()
        del_idx = df_index.loc[df_index['address']==address].index
        df_index = df_index.drop(del_idx)
        
        self._write_fields({'index':df_index})
        
        return None
    
    def _create_group_by_copy(self,source,target,groups,**kwargs):
        
        mode = kwargs.pop('mode','a')
        
        # Boolean that signals if data was written
        success = False
        
        # Open file and check if group already exists
        with h5py.File(self._hdf5_path,  'a') as hdf5_file:
            
            if target in hdf5_file:
                print('Target group already exists.\n')
            
                if mode == 'w':
                    print('Target group is deleted and rewritten.\n')
                    self.delete_group(target)
                    
                    # copy soecified groups in source to target
                    for group in groups:
                        hdf5_file.copy(source+'/'+group,target+'/'+group)
                    
                else:
                    print('No data is written to group. Pass mode="overwrite" \n to overwrite existing data.')
            else:    
                # copy soecified groups in source to target
                for group in groups:
                    hdf5_file.copy(source+'/'+group,target+'/'+group)
                
                success = True
            
            return success
        
    # def write_to_mp4(self,video_name,folder_path,**kwargs):
    #     """
    #     Writes a video sequence already stored in the hdf5-file under the name
    #     video_name to an mp4-file. Video will be saved in the folder specified 
    #     in folder_path under the file name <video_name>.mp4
        
    #     Parameters
    #     ----------
    #     video_name : str
    #         name under which the video is stored in the hdf5-file. If a 
    #         postprocessed version of the video is to be addressed, then pass
    #         'video_name/filtered' for example, or 'video_name/gradient'
    #     folder_path : str
    #         Path to the folder where the video should be saved. 

            
    #     Returns
    #     -------
    #     None.
    #     """
        
    #     print('''For annotation purposes it is highly recommended to export the 
    #           video sequence as png-images via write_to_png(). mp4 performs
    #           interpolations between frames that to not reflect the original
    #           sensor image and can deaviate from it.''')
        
    #     fs = kwargs.pop('fs',8)
    #     appendix = kwargs.pop('appendix','')
        
    #     address = 'videos/' + video_name

    #     # Load specified video sequence
    #     df_video = pd.read_hdf(self._hdf5_path, address + '/data' + appendix)
        
    #     # Load size information
    #     (w,h) = self.load_size(video_name)

    #     # For writing to an image, only pixel values are needed
    #     pixel_cols = df_video.columns[0:w*h]
    #     df_video = df_video[pixel_cols]
        
                        
    #     # Initialize video writer
        
    #     video_name = video_name.replace('/','_')
        
    #     video = cv2.VideoWriter(folder_path + '/' + video_name + '.mp4',
    #                             cv2.VideoWriter_fourcc(*'H264'), 
    #                             fs, (w,h), isColor = True)  
                
    #     # Get colormap
    #     cmap = mpl.cm.get_cmap('plasma')      
        
    #     # Loop over all frames and write to mp4
    #     for i in df_video.index:
            
    #         img = df_video.loc[i].values.reshape(h,w)
            
    #         img = ( img - img.min() ) / (img.max() - img.min())
            
    #         # img = img  / img.max()
            
    #         RGBA = (cmap(img)*255).astype('uint8')
            
    #         # RGBA = cmap(img)
            
    #         BGR = cv2.cvtColor(RGBA, cv2.COLOR_RGB2BGR)
            
    #         video.write(BGR)


    #     cv2.destroyAllWindows()
    #     video.release()
       
    #     return None
    
    
    # def write_to_png(self,video_name,folder_path,**kwargs):
    #     """
    #     Writes a video sequence already stored in the hdf5-file under the name
    #     video_name to png-images frame by frame. The files will be saved 
    #     in a folder with the name <video_name> in the folder specified in
    #     folder_path. The png-files will be named according to their frame 
    #     number.
        
    #     Parameters
    #     ----------
    #     video_name : str
    #         name under which the video is stored in the hdf5-file. If a 
    #         postprocessed version of the video is to be addressed, then pass
    #         'video_name/filtered' for example, or 'video_name/gradient'
    #     folder_path : str
    #         Path to the folder where the video should be saved. 

            
    #     Returns
    #     -------
    #     None.
    #     """
        

    #     appendix = kwargs.pop('appendix','')
        
    #     address = 'videos/' + video_name

    #     # Load specified video sequence
    #     df_video = pd.read_hdf(self._hdf5_path, address + '/data' + appendix)
        
    #     # Load size information
    #     (w,h) = self.load_size(video_name)

    #     # For writing to an image, only pixel values are needed
    #     pixel_cols = df_video.columns[0:w*h]
    #     df_video = df_video[pixel_cols]
        
    #     # Look up maximum and minimum for scaling
    #     dK_max  = df_video.max().max()
    #     dK_min  = df_video.min().min()
        
    #     # Create folder in path with name of video
    #     video_name = video_name.replace('/','_')
        
    #     save_path = folder_path + '/' + video_name
        
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
        
    #     for i in df_video.index:
    #         img = df_video.loc[i].values.reshape((h,w))
            
    #         # file_name = video_name + '_' + str(i) + '.png'
            
    #         file_name = str(i) + '.png'
            
    #         img = ( img - dK_min ) / (dK_max - dK_min)
            
    #         # scipy.misc.toimage(img, cmin=0, cmax=1).save(folder_path + '/' \
    #         #                                                + file_name)
    #         matplotlib.image.imsave(save_path + '/' + file_name, img)
    #     return None
    