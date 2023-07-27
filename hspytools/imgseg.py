# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:31:05 2023

@author: Rehmer
"""

import os


import numpy as np

import matplotlib

import matplotlib.pyplot as plt
from pathlib import Path

    
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
        df_dist = dist_from_center(img_temp)
     
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

        
def dist_from_center(img):
    
    data = []
    
    # replace all nan with zeros for this calculation
    img_nonan = np.nan_to_num(img)
    
    center = ndimage.center_of_mass(abs(img_nonan))
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dist = np.sqrt((x-center[1])**2+(y-center[0])**2)
            data.append([x,y,img[y,x],dist])
    
    df = pd.DataFrame(data = data, columns = ['x','y','p','d'])
    
    
    return df

def hierarch_clust(img,**kwargs):
    '''
    Performs hierarchical clustering on an image given as numpy array
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    dK_std_lim = kwargs.pop('dK_std_lim',0.01)
    dK_lim = kwargs.pop('dK_lim',1)
    
    n_clusters_lim = kwargs.pop('n_clusters_lim',20)
        
    # Calculate distance from center
    df = dist_from_center(img)
   
    # A boolean that controls when to stop
    iterate = True
    
    # Number of clusters to start with
    n_clusters = 3
   
    while iterate == True:
        
        # Perform hierarchical clustering
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                                       affinity='euclidean',
                                                       linkage='ward',
                                                       compute_distances=True)
       
        # Cluster with non nan values
        idx_nonan = df.loc[~df['p'].isna()].index
        
        hierarchical_cluster.fit(df.loc[idx_nonan,['d','p']])
       
        df.loc[idx_nonan,'cluster'] = hierarchical_cluster.labels_ 
        
        # distances in cluster space
        # df.loc[idx_nonan,'clust_dist'] = hierarchical_cluster.distances_
        
        # Evaluate some statistics on found clusters
        df_stat = pd.DataFrame(columns = ['mean_p','std_p','mean_d','std_d']) 
        
        for c in df.loc[idx_nonan,'cluster'].unique():
            mean_p = df.loc[df['cluster']==c,'p'].mean()
            std_p = df.loc[df['cluster']==c,'p'].std()
            
            mean_d = df.loc[df['cluster']==c,'d'].mean()
            std_d = df.loc[df['cluster']==c,'d'].std()
            
            df_stat.loc[int(c)] = [mean_p,std_p,mean_d,std_d]
                        
        # std_rel = abs(df_stat.loc[idx,'std']/df_stat.loc[idx,'mean'])
        dK_std = abs(df_stat['std_p']/df_stat['std_d'])
        dK = df_stat['mean_p'].diff().abs().min()
        
        
        # Do not consider clusters with zero standard deviation (=background)
        dK_std = dK_std.loc[~dK_std.isna()]
        # dK = 
        
        if all(dK_std <= dK_std_lim) or dK < dK_lim or\
            (n_clusters>=n_clusters_lim):
            iterate = False
        else:
            n_clusters = n_clusters + 1
        
        
    return df,df_stat
    


def otsu_thresholding(img):
    
    def compute_otsu_criteria(im, th):
        # create the thresholded image
        thresholded_im = np.zeros(im.shape)
        thresholded_im[im >= th] = 1

        # compute weights
        nb_pixels = im.size
        nb_pixels1 = np.count_nonzero(thresholded_im)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return np.inf

        # find all pixels belonging to each class
        val_pixels1 = im[thresholded_im == 1]
        val_pixels0 = im[thresholded_im == 0]

        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

        return weight0 * var0 + weight1 * var1
    
    # Otsu thresholding only works for positive pixel intensities
    if np.min(img) < 0:
        offset = abs(np.min(img))
    else:
        offset = 0

    img_off = img + offset
    
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = range(int(np.max(img_off))+1)
    criterias = [compute_otsu_criteria(img_off, th) for th in threshold_range]
    
    # best threshold is the one minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]
    
    # Compensate for offset
    best_threshold = best_threshold - offset
    
    # Threshold image
    img_below = img.copy()
    img_above = img.copy()
    img_below[img_below>best_threshold] = np.nan
    img_above[img_above<best_threshold] = np.nan
    
    return img_below,img_above