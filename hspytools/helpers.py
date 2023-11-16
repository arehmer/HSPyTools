# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:15:16 2023

@author: Rehmer
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
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
        
        self.u = u

    def derive_polygon(self,df):
        '''
        Derive a quadrilateral polygon from the dataset, which is then used
        to define the limits of the parameter varying part of the model
        '''
    
        # Get the vertices of the quadrilateral polygon by sorting the data
        # and looping through it looking for the most extreme points per
        # measurement series (Ta)
        
        v1 = df.loc[df['Ta']==df['Ta'].min()].min()[self.u]
        v2 = df.loc[df['Ta']==df['Ta'].min()].max()[self.u]
        v3 = df.loc[df['Ta']==df['Ta'].max()].max()[self.u]
        v4 = df.loc[df['Ta']==df['Ta'].max()].min()[self.u]

        # convert vertices to DataFrame
        V = [v1,v2,v3,v4]
        polygon = []
        
        for v in range(4):
            polygon.append(pd.DataFrame(data = [V[v]],
                                  columns=self.u,
                                  index = ['v'+str(v)]))
        
        polygon = pd.concat(polygon)
        
        self.polygon = polygon
        
        return polygon

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
    
        u0 = 'Ud'
        u1 = 'Tamb0'
        
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
            a = dv[u0]/dv[u1]
            
            # Write line segment through point as line equation Ud = d
            d = pnt[u0] - v0[u0]
            
            # Point of interesection 
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
        
        df_qpoly['d_Tamb0'] = abs(df_qpoly['Tamb0'] - df_pnt['Tamb0'].item())
        
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
            u0 = 'Ud'
            u1 = 'Tamb0'
            
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
                a = dv[u0]/dv[u1]
                
                # Write line segment through point as line equation Ud = d
                d = pnt[u0] - v0[u0]
                
                # Point of interesection 
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
            
            df_qpoly['d_Tamb0'] = abs(df_qpoly['Tamb0'] - df_pnt['Tamb0'].item())
            
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


