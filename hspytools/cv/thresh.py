# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:31:05 2023

@author: Rehmer
"""

import numpy as np

                      
class Otsu():
    
    def __init__(self,**kwargs):
        """
        q: In Heimann TPArray images Otsu tends to set the treshold to high,
           i.e. pixels representing the edges of an object tend to be 
           classified as background. To counter that effect the parameter q
           can be used to lower the threshold. Instead of taking the pixel
           value that minimises Otsu's criteria as the threshold, the 
           q-th percentile of all calculated criterias is taken to be the 
           threshold.
           As a result, pixels that would have fallen under the threshold
           as background do now pass the lower threshold and are assigned
           to foreground pixels.
        
        """
                
        self.q = kwargs.pop('q',100)
    
    def otsu_thresholding(self,img):
        
        # Otsu thresholding only works for positive pixel intensities
        if np.min(img) < 0:
            offset = abs(np.min(img))
        else:
            offset = 0
        
        # Add offset to make everyting greater equal zero
        img_off = img + offset
        
        # Consider only non nan values 
        img_off = img_off[~np.isnan(img_off)].flatten()
        # testing all thresholds from 0 to the maximum of the image
        threshold_range = range(int(np.nanmin(img_off)+1),
                                int(np.nanmax(img_off))+1)
        criterias = [self._1d_otsu(img_off, th) for th in threshold_range]
        
        if self.q == 100:
            # best threshold is the one minimizing the Otsu criteria
            best_threshold = threshold_range[np.argmin(criterias)]
        else:
            # Tweak on Otsu's method to lower the threshold systematically
            criterias = np.array(criterias)
            
            # First get rid of all criteria value above the optimal threshold
            # according to Otsu
            criterias = -criterias[0:np.argmin(criterias)+1]
            
            # Then calculate the q-th percentile
            percentile = np.percentile(criterias,self.q)
            
            # Find the value closest to that percentile and define it as the 
            # threshold
            best_threshold = threshold_range[np.argmin(abs(criterias-percentile))]
            
        # Compensate for offset
        best_threshold = best_threshold - offset
        
        # Threshold image
        img_below = img.copy().astype(float)
        img_above = img.copy().astype(float)
        img_below[img_below>best_threshold] = np.nan
        img_above[img_above<best_threshold] = np.nan
        
        return img_below,img_above
    
    
    def _1d_otsu(self,img,th):
        
        # create the thresholded image
        thresholded_img = np.zeros(img.shape)
        thresholded_img[img >= th] = 1

        # compute weights
        nb_pixels = img.size
        nb_pixels1 = np.count_nonzero(thresholded_img)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one of the classes is empty, eg all pixels are below or above the
        # threshold, that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return np.inf

        # find all pixels belonging to each class
        val_pixels1 = img[thresholded_img == 1]
        val_pixels0 = img[thresholded_img == 0]

        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

        return weight0 * var0 + weight1 * var1
    
    # def _2d_otsu(self,img,th):
        
    #     # create the thresholded image
    #     thresholded_im = np.zeros(im.shape)
    #     thresholded_im[im >= th] = 1

    #     # compute weights
    #     nb_pixels = im.size
    #     nb_pixels1 = np.count_nonzero(thresholded_im)
    #     weight1 = nb_pixels1 / nb_pixels
    #     weight0 = 1 - weight1

    #     # if one of the classes is empty, eg all pixels are below or above the
    #     # threshold, that threshold will not be considered
    #     # in the search for the best threshold
    #     if weight1 == 0 or weight0 == 0:
    #         return np.inf

    #     # find all pixels belonging to each class
    #     val_pixels1 = im[thresholded_im == 1]
    #     val_pixels0 = im[thresholded_im == 0]

    #     # compute variance of these classes
    #     var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    #     var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    #     return weight0 * var0 + weight1 * var1    
    

    
    
        
        
        
        
        