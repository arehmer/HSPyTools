# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:15:16 2023

@author: Rehmer
"""

import pickle as pkl
import pandas as pd
import numpy as np
from pathlib import Path
import ctypes
from scipy.interpolate import LinearNDInterpolator as lNDI

from ..tpiles.tparray import TPArray

class HTPAdGUI_FileReader():
    
    def __init__(self,width,height):
        
        self.width = width
        self.height = height
        
        
        # Depending on the sensor size, the content of the bds-file is
        # organized as follows
        self.tparray = TPArray(width,height)
        # data_order = ArrayType.get_serial_data_order()
        
        
    def read_htpa_video(self,path):
        
        # Check if path is provided as pathlib path or string
        if isinstance(path,Path):
            path = path.as_posix()
        
        # Check the last 4 characters in path for file extension
        extension = path[-4::]
        
        if extension.casefold() == ('.txt').casefold():
            df_video = self._import_txt(path) 
        elif extension.casefold() == ('.bds').casefold():
            df_video = self._import_bds(path)
        else:
            print('File extension not recognized.')
            return None
        
        # print("""Pixel array is flipped vertically otherwise mirrored
        #       image is displayed""")
              
        # df_video = self._flip(df_video)
        
        return df_video
    
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
        txt_content.index = range(1,len(txt_content)+1)
        txt_content.index.name = 'image_id'
                
        return txt_content
    
    def _import_bds(self,bds_path,**kwargs):
        
        # open file and save content byte by byte in list
        bds_content = []
        
        with open(bds_path, "rb") as f:
            
            # Skip header, i.e. first 32 bytes
            f.read(31)
            
            # Read two bytes at a time 
            while (LSb := f.read(2)):
                
                # and combine in LSb fashion
                bds_content.append(int.from_bytes(LSb, 
                                                  byteorder='little'))
                
        
        # Cast the data to a DataFrame of appropriate size
        columns = self.tparray.get_serial_data_order()
        
        bds_content = (np.array(bds_content)).reshape(-1,len(columns))
        bds_content = pd.DataFrame(data=bds_content,
                                   columns = columns)
        
        bds_content.index = range(1,len(bds_content)+1)
        bds_content.index.name = 'image_id'
        

        
    
    def _flip(self,df_video):

        w = self.width
        h = self.height
        
        pix_cols = df_video.columns[0:w*h]
        
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
        polygon = self.polygon
        
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
            
            # If intersection is not on the edge
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


