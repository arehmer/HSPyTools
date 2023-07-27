# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:31:05 2023

@author: Rehmer
"""

import os

import pandas as pd
import numpy as np

import matplotlib

import matplotlib.pyplot as plt
from pathlib import Path

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from ..tpiles.tparray import TPArray
import hsfit.imgseg.helpers as hlp


    
class Seg():
    
    def __init__(self,w,h,**kwargs):
        
        self.w = w
        self.h = h
        
    def write_to_png(self,title,file_name):
        """
        Save the last image and the segmented image as a png for later 
        inspection

        Parameters
        ----------
        title : string
            A string that will be used as the title of the image
        file_name: string
            A string that will be used as the name of the png-file

        Returns
        -------
        None.

        """
        
        img = self.img
        img_seg = self.img_seg
        
        # Use Agg to not show figure
        matplotlib.use('Agg')
        
        # Get maximum and minimum of data for scaling purposes
        vmin = img.min()
        vmax = img.max()
        
        # Plot as visual control
        plt.ioff()
        fig,ax = plt.subplots(1,2)
        fig.suptitle(title)
        im1 = ax[0].imshow(img,vmin=vmin,vmax=vmax)
        im2 = ax[1].imshow(img_seg,vmin=vmin,vmax=vmax)
    

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        
        save_path = Path.cwd() / 'img_seg'
        
        # Check if directory for saving clustering results exists
        if not save_path.is_dir():
            save_path.mkdir()
        
        # Check if file exists in folder       
        file_path = save_path / file_name
        
        if file_path.exists():
            os.remove(file_path)
        
                
        plt.savefig(file_path)
        plt.close(fig)
        
        # Change backend to show figures again
        matplotlib.use('Qt5Agg')
        
    
    def _choose_foreground(self,img_below,img_above,dT):
        
        if  dT >= 5:
            img_thresh = img_above  
        else:
            # In this case the algorithm does not work robustly
            # ask user to decide
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(img_below)
            ax[0].set_title('below threshold')
            ax[1].imshow(img_above)
            ax[1].set_title('above threshold')
            fig.show()
            plt.pause(10E-3)
            
            print("""User input needed. Choose which image represents the 
                  object.""")
            choice = input("""Enter 'b' for the image below the threshold 
                           or 'a' for the image above the threshold:   """)            
            
            plt.close(fig)
            
            if choice == 'b':
                img_thresh = img_below
            elif choice == 'a':
                img_thresh = img_above
                
            return img_thresh
        


class ClustSeg(Seg):
    
    def __init__(self,w,h,**kwargs):
        
        super(ClustSeg,self).__init__(w,h)
  
        self.hierarch_clust = hierarch_clust

    def segment(self,img,**kwargs):

        Ta = kwargs.pop('Ta',np.nan)
        To = kwargs.pop('To',np.nan)        
        
        # Crop the image by 3 pixels to get rid of corners.
        img_crop = img[3:self.h-3,3:self.w-3]

        # Threshold image
        img_below, img_above= otsu_thresholding(img_crop)
        
        # Depending on difference between ambient and object temperature 
        # the object appears below or above the threshold in the image
        img_thresh = self._choose_foreground(img_below,img_above,To-Ta)
         
        # To preserve shape create an NaN array of the original shape and 
        # insert the thresholded image
        img_temp = np.zeros((self.h,self.w))*np.nan
        img_temp[3:self.h-3,3:self.w-3] = img_thresh
        
        # CLuster the thresholded image
        df_clust,df_clust_stat = self.hierarch_clust(img_temp,
                                                     std_lim=0.01)
        
        # Depending on the relation between ambient temperature Ta and 
        # object temperature the object is represented by the most 
        # or least intensive pixels.
        if  To > Ta:
            c_idx = df_clust_stat['mean_p'].idxmax()  
        elif To < Ta:
            c_idx = df_clust_stat['mean_p'].idxmin()  
        elif To == Ta:
            c_idx = df_clust_stat['mean_d'].idxmin()  
        
        # Select pixels belonging to the cluster c_idx 
        idx_sel = df_clust.loc[df_clust['cluster']==c_idx].index
        
        # Create a new image containing only the selected pixels
        img_seg = np.zeros((self.h,self.w))*np.nan
        img_seg = img_seg.flatten()
        img_seg[idx_sel] = df_clust.loc[idx_sel,'p'].values.flatten()
        
        # reshape
        img_seg = img_seg.reshape((self.h,self.w))

        # Save image and segmented image as attributes
        self.img = img
        self.img_seg = img_seg
        
        return img_seg

class RegionSeg(Seg):
    
    def __init__(self,w,h,**kwargs):
        
        super(RegionSeg,self).__init__(w,h)

        self.r = kwargs.pop('r',3)
  
        # self.hierarch_clust = hierarch_clust
        
    def segment(self,img,**kwargs):
        
        Ta = kwargs.pop('Ta',np.nan)
        To = kwargs.pop('To',np.nan)
        
        # Crop the image by 3 pixels to get rid of corners.
        img_crop = img[3:self.h-3,3:self.w-3]
        
        # Threshold image
        img_below, img_above= otsu_thresholding(img_crop)
        
        # Depending on difference between ambient and object temperature 
        # the object appears below or above the threshold in the image
        img_thresh = self._choose_foreground(img_below,img_above,To-Ta)
        

        # To preserve shape create an NaN array of the original shape and 
        # insert the thresholded image
        img_temp = np.zeros((self.h,self.w))*np.nan
        img_temp[3:self.h-3,3:self.w-3] = img_thresh

        
        # Calculate distance of each pixel from center of gravity
        df_dist = hlp.dist_from_center(img_temp)
     
        # Select pixels inside specified radius
        idx_sel = df_dist.loc[(df_dist['d']<=self.r) &\
                              ~(df_dist['p']).isna()].index
        
        # Create a new image array containing only the selected pixels
        img_seg = np.zeros((self.h,self.w))*np.nan
        img_seg = img_seg.flatten()
        img_seg[idx_sel] = df_dist.loc[idx_sel,'p'].values.flatten()
        
        # reshape
        img_seg = img_seg.reshape((self.h,self.w))
        
        # Save image and segmented image as attributes
        self.img = img
        self.img_seg = img_seg
        
        return img_seg
        
        
    