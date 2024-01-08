# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:15:16 2023

@author: Rehmer
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import os
import openpyxl 
import matplotlib.pyplot as plt
# import imageio_ffmpeg



# import matplotlib.pyplot as plt
# from scipy.interpolate import LinearNDInterpolator as lNDI

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
            img = df_video.loc[i].values.reshape(npsize)

            
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
               
 
class Byte_Stream_Converter():
    def __init__(self,width,height):
        
        self.width = width
        self.height = height
        
    def bytes_to_img(self,byte_stream):
        
             
        # Loop over all bytes and combine MSB and LSB
        idx = np.arange(0,len(byte_stream),2)
        
        img = np.zeros((1,self.width*self.height))
        j=0
        
        for i in idx:    
            img[0,j] = int.from_bytes(byte_stream[i:i+2], byteorder='little')
            j = j+1
            
        img = img.reshape((self.height,self.width))
        img = np.flip(img,axis=1)
        
        return img
        

class QuadriPolygon():
    
    def __init__(self,u):
        
        self.u = ['input1','input2']
        
        # Make sure Tamb0 is always the first input
        if 'Tamb0' in u[0]:
            self.u[0] = u[0]
            self.u[1] = u[1]
        else:
            self.u[0] = u[1]
            self.u[1] = u[0]
            

        
    def derive_polygon(self,df):
        '''
        Derive a quadrilateral polygon from the dataset, which is then used
        to define the limits of the parameter varying part of the model
        '''
    
        # Get the vertices of the quadrilateral polygon by sorting the data
        # and looping through it looking for the most extreme points per
        # measurement series (Ta)
        # v1 = df.loc[df['Ta']==df['Ta'].min()].min()[self.u]
        # v2 = df.loc[df['Ta']==df['Ta'].min()].max()[self.u]
        # v3 = df.loc[df['Ta']==df['Ta'].max()].max()[self.u]
        # v4 = df.loc[df['Ta']==df['Ta'].max()].min()[self.u]


        # convert vertices to DataFrame
        # V = [v1,v2,v3,v4]
        
        V = self._get_vertives(df)
        
        polygon = []
        
        for v in range(4):
            polygon.append(pd.DataFrame(data = [V[v]],
                                  columns=self.u,
                                  index = ['v'+str(v)]))
        
        polygon = pd.concat(polygon)
        
        self.polygon = polygon
        
        return polygon
    
    def _get_vertives(self,df):
        """
        Derives vertices of polygon from the data. 
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        tol = 0.05
        
        T = self.u[0]
        U = self.u[1]
        
        # v0
        Tamb0 = df.loc[df[T]==df[T].min(),T].min()
        Ud0 = df.loc[df[T]<=(1+tol)*Tamb0,U].min()
        
        v0 = pd.Series(data=[Tamb0,Ud0],index=[T,U],name='v0')
        
        # v1
        Tamb0 = df.loc[df[T]==df[T].min(),T].min()
        Ud1 = df.loc[df[T]<=(1+tol)*Tamb0,U].max()
        
        v1 = pd.Series(data=[Tamb0,Ud1],index=[T,U],name='v1')
        
        # v2
        Tamb2 = df.loc[df[T]==df[T].max(),T].max()
        Ud2 = df.loc[df[T]>=(1-tol)*Tamb2,U].max()
        
        v2 = pd.Series(data=[Tamb2,Ud2],index=[T,U],name='v2')

        # v3
        Ud3 = df.loc[df[T]>=(1-tol)*Tamb2,U].min()
        
        v3 = pd.Series(data=[Tamb2,Ud3],index=[T,U],name='v3')        
        
        V = [v0,v1,v2,v3]
        
        return V
        
    def in_polygon(self,df_pnt):
        '''
        This method checks if a point is inside or outside the quadrilateral 
        polygon that defines the area of the input space where data was 
        available for estimating the parameter variation
        
        THIS METHOD WORKS ONLY IF THE POLYGON IS A RECTANGLE
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        
        # dataframe to series
        df_pnt = df_pnt.squeeze()
        
        # Boolean
        in_poly = False

        # get polygon
        polygon = self.polygon.copy()
        
        # Extract scheduling variables from df_pnt
        df_pnt = df_pnt[self.u]
        
        # Debugging
        # df_pnt.iloc[0]['Ud'] = 500
        # df_pnt.iloc[0]['Tamb0'] = 300
        
        # Edges of the polygon
        E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
        
        # direction of the ray 
        r = df_pnt.copy()
        r[self.u[0]] = 1
        r[self.u[1]] = 1
        
        intersect = []
        
        # Project point on all edges
        for e in E:
            

           
            v0 = polygon.loc[e[0]]
            v1 = polygon.loc[e[1]]

            # calculate intersection of the two lines p + k0*r and v0 + k1*(v1-v0)
            # by solving system of linear eqations
            A = np.array([[r[self.u[0]],-(v1-v0)[self.u[0]]],
                         [r[self.u[1]],-(v1-v0)[self.u[1]]]])
            b = (v0-df_pnt).values.flatten()
            
            k = np.linalg.solve(A.astype(float), b.astype(float))
            
            # k0 is parameter of the ray from the point in question
            # k1 is parameter on the edge 
            
            # if k0 is negative the ray does not intersect the line
            # if k1 is negative or larger than one, the ray does not
            # intersect the line
            if k[0]<-10E-6 or k[1]>1 or k[1]<-10E-6:
                intersect.append(False)
            else:
                intersect.append(True)
            

        # if the number of intersections is odd, the point is inside the polygon
        if sum(intersect)%2 == 1:
            in_poly = True
          
        return in_poly   
            
    def pnt_on_edge(self,df_pnt):
        '''
        Calculates the closest point on the edge of the quadrilateral 
        polygon to a given point   
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        

        
        # dataframe to series
        pnt = df_pnt.squeeze()
        
        # Extract scheduling variables from df_pnt
        pnt = pnt[self.u]
        
        # Edges of the polygon
        # E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
              
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
    
        u0 = self.u[1]#'Ud'
        u1 = self.u[0]#'Tamb0'
        
        # Loop over edges v1,v2 and v3,v0 first
        E = [('v1','v2'),('v3','v0')]
        
        
        # Project point on edges ('v1','v2'),('v3','v0')
        for e in E:
            
            # Get quadrilateral polygon
            df_qpoly = self.polygon.copy()
           
            #Get end points of line segment
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
        
            # Check if point can be projected on line segment
            
            # Check temperature condition first!
            # if min(v0[u1], v1[u1]) < pnt[u1] < max(v0[u1], v1[u1]):
                
            # Write line segment as line equation Tamb0 = a*Ud
            dv = v1 - v0
            a = dv[u1]/dv[u0]
            
            # Write line segment through point as line equation Tamb0 = d
            d = pnt[u1] - v0[u1]
            
            # Point of interesection 
            isect = pd.Series(data=[d/a + v0[u0] ,a*(d/a) + v0[u1]],
                                 index = [u0,u1])
            
            # Save intersection point
            intersect.append(isect)
        
            # Distance between intersection and point
            dist.append(np.linalg.norm(pnt-isect))
        
        if len(dist)>0:
            
            # Find minimal distance
            idx_min = np.array(dist).argmin()
            
            # Get the corresponding intersection point
            isect = intersect[idx_min]
            
            # check if that point is in the polygon
            if self.in_polygon(isect+0.001*isect) or \
                self.in_polygon(isect-0.001*isect):
                    
                    
                    # Convert from series to dataframe
                    df_isect = pd.DataFrame(data=[isect.values],
                                            columns=isect.index,
                                            index=df_pnt.index)
                    
                    return df_isect
                
            
        # If no intersection point could be returned, project point on   
        # edge ('v0','v1') and ('v2','v3')
        E = [('v0','v1'),('v2','v3')]
        
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
                
        # Project point on edges ('v1','v2'),('v3','v0')
        for e in E:
            # Get quadrilateral polygon
            df_qpoly = self.polygon.copy()
           
            #Get end points of line segment
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
        
            # Check if point can be projected on line segment
        
            # if min(v0[u0], v1[u0]) < pnt[u0] < max(v0[u0], v1[u0]):
                
            # Write line segment as line equation Ud = a*Tamb0
            dv = v1 - v0
            
            # Write line segment through point as line equation Ud = d
            d = pnt[u0] - v0[u0]
            
            # In case that both vertices have same Ta_coordinate, df[u1] 
            # will become zero. Treat this exception:            
            if dv[u1]==0:
                isect = pd.Series(data=[0+v0[u1] ,d + v0[u0]],
                                     index = [u1,u0])
            else: 
                a = dv[u0]/dv[u1]
                isect = pd.Series(data=[d/a + v0[u1] ,a*(d/a) + v0[u0]],
                                     index = [u1,u0])
            
            # Save intersection point
            intersect.append(isect)
        
            # Distance between intersection and point
            dist.append(np.linalg.norm(pnt-isect))
        
        if len(dist)>0:
            idx_min = np.array(dist).argmin()
            
            # Get the corresponding intersection point
            isect = intersect[idx_min]
            
            # Check if intersection is in polygon 
            if self.in_polygon(isect+0.001*isect) or \
                self.in_polygon(isect-0.001*isect):
                    # Convert from series to dataframe
                    df_isect = pd.DataFrame(data=[isect.values],
                                            columns=isect.index,
                                            index=df_pnt.index)
                    
                    return df_isect
                 
        # In the case that projection on the edge was not possible, 
        # find the closest vertex      
        
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
        
        df_qpoly['d_Tamb0'] = abs(df_qpoly[u1] - df_pnt[u1].item())
        
        # Sort by difference and keep the first two
        df_qpoly = df_qpoly.sort_values(by='d_Tamb0')
        
        df_qpoly = df_qpoly.iloc[0:2]
        
        # Among these two vertices find the closest            
        vert = [df_qpoly.loc[v] for v in df_qpoly.index]
        
        # Find vertex closest 
        d =  [np.linalg.norm(pnt[self.u]-v[self.u]) for v in vert]
                   
        d = np.array(d)
        
        # Distance between closest vertice and point
        dist.append(min(d))
        
        # Save intersection point
        isect = (vert[d.argmin()]).copy()
        intersect.append(isect[self.u])
        
        df_isect = pd.DataFrame(data=[isect.values],
                                columns=isect.index,
                                index=df_pnt.index)
         

         
        return df_isect
            
    
             
    def pnt_on_edge_old(self,df_pnt):
        '''
        Calculates the closest point on the edge of the quadrilateral 
        polygon to a given point   
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        

        
        # dataframe to series
        pnt = df_pnt.squeeze()
        
        # Extract scheduling variables from df_pnt
        pnt = pnt[self.u]
        
        # Edges of the polygon
        E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
              
        # list for storing distances in
        dist = []
        
        # List for storing intersection with polygon in
        intersect = []
        

     
        
        # Project point on all edges
        for e in E:
            
            # Get quadrilateral polygon
            df_qpoly = self.polygon.copy()
           
            #Get end points of line segment
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
            

            # Check if point can be projected on line segment
            u0 = 'Ud_pv'
            u1 = 'Tamb0_pv'
            
            # Check temperature condition first!
            if min(v0[u1], v1[u1]) < pnt[u1] < max(v0[u1], v1[u1]):
                
                # Write line segment as line equation Tamb0 = a*Ud
                dv = v1 - v0
                a = dv[u1]/dv[u0]
                
                # Write line segment through point as line equation Tamb0 = d
                d = pnt[u1] - v0[u1]
                
                # Point of interesection 
                isect = pd.Series(data=[d/a + v0[u0] ,a*(d/a) + v0[u1]],
                                     index = [u0,u1])
                
                # Check if intersection is in polygon 
                if self.in_polygon(isect+0.001*isect) or \
                    self.in_polygon(isect-0.001*isect):
                    
                        dd = np.linalg.norm(pnt-isect)    
                        
                        # check if any vertice is closer in Ta-direction than
                        # intersection point
                        # if v0[u1] - pnt[u1] < dd:
                        #     isect = v0
                        # elif v1[u1] - pnt[u1] < dd:
                        #     isect = v1
                        
                        # Save intersection point
                        intersect.append(isect)
                    
                        # Distance between intersection and point
                        dist.append(np.linalg.norm(pnt-isect))
            
            elif min(v0[u0], v1[u0]) < pnt[u0] < max(v0[u0], v1[u0]):
                
                # Write line segment as line equation Ud = a*Tamb0
                dv = v1 - v0
                # Write line segment through point as line equation Ud = d
                d = pnt[u0] - v0[u0]
                
                # In case that both vertices have same Ta_coordinate, df[u1] 
                # will become zero. Treat this exception:
                if dv[u1]==0:
                    isect = pd.Series(data=[0+v0[u1] ,d + v0[u0]],
                                         index = [u1,u0])
                
                else: 
                    a = dv[u0]/dv[u1]
                    isect = pd.Series(data=[d/a + v0[u1] ,a*(d/a) + v0[u0]],
                                         index = [u1,u0])
                
                
                # Check if intersection is in polygon 
                if self.in_polygon(isect+0.001*isect) or \
                    self.in_polygon(isect-0.001*isect):
                        
                        dd = np.linalg.norm(pnt-isect)    
                        
                        # check if any vertice is closer in Ta-direction than
                        # intersection point
                        if abs(v0[u1] - pnt[u1]) < dd:
                            isect = v0
                        elif abs(v1[u1] - pnt[u1]) < dd:
                            isect = v1
                        
                        # Save intersection point
                        intersect.append(isect)

                        # Distance between intersection and point
                        dist.append(np.linalg.norm(pnt-isect))
                
                

    
                
                        
        # If intersection with line is not possible, find closest
        # vertice
        if len(intersect)==0:
            
            df_qpoly['d_Tamb0'] = abs(df_qpoly[u1] - df_pnt[u1].item())
            
            # Sort by difference and keep the first two
            df_qpoly = df_qpoly.sort_values(by='d_Tamb0')
            
            df_qpoly = df_qpoly.iloc[0:2]
            
            # Among these two vertices find the closest            
            vert = [df_qpoly.loc[v] for v in df_qpoly.index]
            
            # Find vertice closest 
            d =  [np.linalg.norm(pnt[self.u]-v[self.u]) for v in vert]
                       
            d = np.array(d)
            
            # Distance between closest vertice and point
            dist.append(min(d))
            
            # Save intersection point
            isect = (vert[d.argmin()]).copy()
            intersect.append(isect[self.u])
                
            
            
            
        # In the end find and return the intersection point with the
        # smallest distance
        # Find minimal distance
        idx_min = np.array(dist).argmin()
        
        # Return the corresponding intersection point
        df_isect = intersect[idx_min]
        
        # Convert from series to dataframe
        df_isect = pd.DataFrame(data=[df_isect.values],
                                columns=df_isect.index,
                                index=df_pnt.index)

        
        return df_isect           
            
    def orth_proj_on_edge(self,df_pnt):
        '''
        Calculates the orthogonal projection of a point on the edge of the 
        quadrilateral polygon to a given point   
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        # Get quadrilateral polygon
        df_qpoly = self.polygon
        
        # dataframe to series
        pnt = df_pnt.squeeze()
        
        # Extract scheduling variables from df_pnt
        pnt = pnt[self.u]
        
        # Edges of the polygon
        E = [('v0','v1'),('v1','v2'),('v2','v3'),('v3','v0')]
        
        #
        orth_dist = []
        intersect = []
        
        # Project point on all edges
        for e in E:
            v0 = df_qpoly.loc[e[0]]
            v1 = df_qpoly.loc[e[1]]
            
            #calculate the orthgonal distance to the edge
            a = v1 - v0
            b = pnt - v0
            
            norm_a = np.linalg.norm(a)
            
            d = abs(np.cross(a,b)/norm_a)
            
            # Calculate intersection
            isect = np.dot(a,b)/norm_a * (a/norm_a) + v0
            
            # If intersection is not on the edge (in polygon)
            if not self.in_polygon(isect):
                # calculate distances to vertices
                d1 = np.linalg.norm(pnt-v0)
                d2 = np.linalg.norm(pnt-v1)
                
                d = min([d1,d2])
                isect = list([v0,v1])[[d1,d2].index(d)]
                
            orth_dist.append(d)
            intersect.append(isect)
        
        # Find minimal distance
        idx_min = np.array(orth_dist).argmin()
        
        # Return the corresponding intersection point
        df_isect = intersect[idx_min]
        
        # Convert from series to dataframe
        df_isect = pd.DataFrame(data=[df_isect.values],
                                columns=df_isect.index,
                                index=df_pnt.index)    
        return df_isect

def pd_interpolate_MI (df_input, df_toInterpolate,col):
    
    #create the function of interpolation
    func_interp = lNDI(points=df_input.index.to_frame().values, 
                       values=df_input[col].values)
    #calculate the value for the unknown index
    df_toInterpolate[col] = func_interp(df_toInterpolate.index.to_frame().values)
    #return the dataframe with the new values
    return pd.concat([df_input, df_toInterpolate]).sort_index()


class LuT:
    
    def __init__(self,**kwargs):
        
        pass
    
    def LuT_from_df(self,df):
        
        
        
        
        self.LuT = df
        
    def LuT_from_csv(self,csv_path,offset):
        
        self.offset = offset
        
        # Import data
        LuT = pd.read_csv(csv_path, sep=',',header=0,index_col = 0)
        
        # Convert column header to int
        LuT.columns = np.array([int(c) for c in LuT.columns])
        
        # Subtract offset
        LuT.index = LuT.index - offset
        
        self.LuT = LuT
    
    def LuT_to_xls(self,xls_path):
        
        # Load LuT
        LuT = self.LuT.copy()
        
        # Initialize writer object
        writer = self._init_xlswriter(xls_path)
        
        # Start a list where columns are in the right order
        columns_ordered = list(LuT.columns)
        
        # Reindex so Ud becomes column
        LuT = LuT.reset_index(drop=False)        
        
        # Add a column vector for the offset-free voltage signal
        LuT['Ud_norm'] = LuT['Ud'] - LuT['Ud'].min() 
        
        # Opening and closing brackets
        LuT['br_o'] = ['{']*len(LuT)
        LuT['br_c'] = ['},']*len(LuT)
        
        columns_ordered = ['Ud','Ud_norm','br_o'] + columns_ordered + ['br_c']
        
        LuT = LuT[columns_ordered]
        
        LuT.to_excel(writer,
                     sheet_name = xls_path.stem,
                     startrow=12,
                     index=False)
        
        # Then write normalized Ud to row 9
        df_V = pd.DataFrame(data=[],
                            columns=np.arange(0,len(LuT),1,dtype=int),
                            index=['V'])
        
        columns_ordered = list(df_V.columns)
                
        df_V.loc['V'] = LuT['Ud_norm'].values.astype(int)
        
        df_V = df_V.astype('str')
        df_V[df_V.columns[:-1]] = df_V[df_V.columns[:-1]]+','
        
        df_V['br_o'] = '{'
        df_V['br_c'] = '};'

        columns_ordered =  ['br_o'] + columns_ordered + ['br_c']
        
        df_V = df_V[columns_ordered]
        
        df_V.to_excel(writer,
                      sheet_name = xls_path.stem,
                      startrow=9,
                      index=True,
                      header=False)
        
        # and Ta to row 10
        df_Ta = pd.DataFrame(data=[],
                            columns=list(self.LuT.columns),
                            index=['Ta'])
        
        columns_ordered = list(df_Ta.columns)
                
        df_Ta.loc['Ta'] = list(self.LuT.columns)
        

        
        df_Ta = df_Ta.astype('str')
        df_Ta[df_Ta.columns[:-1]] = df_Ta[df_Ta.columns[:-1]]+','

        
        df_Ta['br_o'] = '{'
        df_Ta['br_c'] = '};'

        columns_ordered =  ['br_o'] + columns_ordered + ['br_c']
        
        df_Ta = df_Ta[columns_ordered]
        
        df_Ta.to_excel(writer,
                       sheet_name = xls_path.stem,
                       startrow=10,
                       index=True,
                       header=False)
        
        writer.save()
            
    def LuT_from_HTPAxls(self,sheet_name):
              
        xls_path = Path('T:/Projekte/HTPA8x8_16x16_32x31/Datasheet/LookUpTablesHTPA.xlsm')
        
        index_col = 0
        usecols = 'A,D:O'
        skiprows = 17
        header = 0
        dtype = 'object'
        
        df = pd.read_excel(xls_path,
                           sheet_name = sheet_name,
                           skiprows = skiprows,
                           index_col = index_col,
                           usecols = usecols,
                           header = header)
        
        # Delete all commata
        df = df.replace(',','',regex=True)
        
        # Cast all columns to int
        df = df.astype(np.int32)
        
        # Rename index
        df.index.name = 'Ud'
        
        # That's it
        self.LuT = df
        
        return None
        
    def inverse_eval_LuT(self,data,Ta_col,To_col):
        
        # Check if index is unique, otherwise loop will exract more than one
        # measurement per loop and algorithm brakes down
        if not data.index.is_unique:
            print('Data index is not unique. reset_index() is applied!')
            data = data.reset_index()
        
        
        for meas in data.index:
            
            LuT_copy = self.LuT.copy()
        
            Ta_meas = data.loc[meas,Ta_col]
            To_meas = data.loc[meas,To_col]
            
            # find the "last" column in old LuT that is smaller than the 
            # measured Ta 
            col_idx = LuT_copy.columns < Ta_meas
            LuT_col = LuT_copy.columns[col_idx][-1]
            
            # get neighbouring column
            Ta_col_n = LuT_copy.columns[LuT_copy.columns.get_loc(LuT_col)+1]
            
            # create a new column by interpolation
            new_col = int(np.round(Ta_meas))
            f = (new_col-LuT_col) / (Ta_col_n-LuT_col)
            LuT_copy[new_col] = LuT_copy[LuT_col] + \
                f*(LuT_copy[Ta_col_n]-LuT_copy[LuT_col])
            
            # Find index of las To in that column that is smaller than the measured To
            Ud_row = LuT_copy.loc[LuT_copy[new_col]<To_meas].index[-1]
            
            # get neighbouring row
            Ud_row_n = LuT_copy.index[LuT_copy.index.get_loc(Ud_row)+1]
            
            # Calculate Ud_est
            dT = LuT_copy.loc[Ud_row_n,new_col] - LuT_copy.loc[Ud_row,new_col]
            dU = Ud_row_n-Ud_row
            
            f =  dU/dT
            
            data.loc[meas,'Ud_LuT'] = Ud_row + \
                (To_meas - LuT_copy.loc[Ud_row,new_col]) * f
                
        return data
        
    def eval_LuT(self,data,Ta_col='Tamb0',Ud_col='Ud'):
        
        LuT = self.LuT
        
        for meas in data.index:
    
            Ta_meas = data.loc[meas,Ta_col]
            Ud = data.loc[meas,Ud_col]
        
            # Find columns and indeces for bilinear interpolation
            col_idx = LuT.columns < Ta_meas
            LuT_col = LuT.columns[col_idx][-1]
            Ta_col_n = LuT.columns[LuT.columns.get_loc(LuT_col)+1]
            
            row_idx = LuT.index < Ud
            Ud_row = LuT.index[row_idx][-1]
            Ud_row_n = LuT.index[LuT.index.get_loc(Ud_row)+1]
            
            
            rect_pnts = LuT.loc[Ud_row:Ud_row_n,[LuT_col,Ta_col_n]]
            
            data.loc[meas,'To_LuT'] = \
                self._get_To(rect_pnts,Ta_meas,Ud)
                
        return data
        
        
    def _get_To(self,points,Ta_meas,Ud_meas):
        
        x0 = points.columns[0]
        x1 = points.columns[1]
        
        y0 = points.index[0]
        y1 = points.index[1]
    
        
        pt1 = (x0,y0,points.loc[y0,x0])
        pt2 = (x0,y1,points.loc[y1,x0])
        pt3 = (x1,y0,points.loc[y0,x1])
        pt4 = (x1,y1,points.loc[y1,x1])
            
        rect_pnts = np.array([pt1,pt2,pt3,pt4])
        
        return self._bilinear_interpolation(Ta_meas,Ud_meas,rect_pnts)
        
    def _bilinear_interpolation(self,x, y, points):
        '''Interpolate (x,y) from values associated with four points.
    
        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.
    
            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0
    
        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    
        # points = sorted(points)               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
    
        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')
    
        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0)
        
    def plot_LuT(self):
        
        LuT = self.LuT
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X = np.array(LuT.columns).reshape((1,-1))
        X = np.repeat(X,len(LuT),axis=0)
        
        Y = np.array(LuT.index).reshape((-1,1))
        Y = np.repeat(Y,len(LuT.columns),axis=1)
        
        Z = LuT.values
        
                
        ax.plot_surface(X,Y,Z,cmap=matplotlib.cm.coolwarm,antialiased=False)
        ax.set_xlabel('Tamb0')
        ax.set_ylabel('Ud')
        ax.set_zlabel('To_pred')
        
        pass
            
    def _init_xlswriter(self,path):
        
        # Load or create workbook
        writer = pd.ExcelWriter(path, engine='openpyxl')
        writer.book.create_sheet(title=path.stem)
        
        return writer