# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:16:48 2023

@author: Rehmer
"""
import numpy as np
import ctypes
import pandas as pd
import json
from pathlib import Path
import struct

class TPArray():
    """
    Class contains hard-coded properties of Thermopile-Arrays relevant
    for reading from Bytestream
    """
    
    def __init__(self,width,height):
        
        
        # Basic attributes
        self._width = width
        self._height = height
        self._size = (width,height)
        self._npsize = (height,width)
        
        # Calculate order of data in bds file
        DevConst = {}
        
        if (width,height) == (8,8):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=1
            DevConst['NROFPTAT']=1
            
            self._package_num = 1
            self._package_size = 262
            self._fs = 160
            self._NETD = 100
            
        elif (width,height) == (16,16):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=2
            DevConst['NROFPTAT']=2
            
            self._package_num = 1
            self._package_size = 780
            self._fs = 70
            self._NETD = 130
        
        elif (width,height) == (32,32):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=4
            DevConst['NROFPTAT']=2
            
            self._package_num = 2
            self._package_size = 1292
            self._fs = 27
            self._NETD = 140
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '32x32.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif (width,height) == (80,64):
            DevConst['NROFBLOCKS']=4
            DevConst['NROFPTAT']=2
            DevConst['ATCaddr']=0
            
            self._package_num = 10
            self._package_size = 1283
            self._fs = 41
            self._NETD = 70
           
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '80x64.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif (width,height) == (60,84):
            DevConst['NROFBLOCKS']=7
            DevConst['NROFPTAT']=2
            DevConst['ATCaddr']= 0
                        
            
            self._package_num = 10
            self._package_size = 1283
            self._fs = 41
            self._NETD = 70
           
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '60x84.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif (width,height) == (120,84):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=6
            DevConst['NROFPTAT']=2
            
            self._package_num = 17
            self._package_size = 1401
            self._fs = 20
            self._NETD = 130
            r_lim = 60
            
            self._mask = self._binary_mask(r_lim) 
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '120x84.json'
            # Load calibration data from file
            self._load_calib_json(path)
            
        elif (width,height) == (60,40):
            DevConst['ATCaddr']=1
            DevConst['NROFBLOCKS']=5
            DevConst['NROFPTAT']=2
            
            self._package_num = 5
            self._package_size = 1159
            self._fs = 47
            self._NETD = 90
            
            self._mask = np.ones(self._npsize)
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '60x40.json'
            # Load calibration data from file
            self._load_calib_json(path)  

        elif (width,height) == (160,120):
            DevConst['ATCaddr'] = 1
            DevConst['NROFBLOCKS'] = 12
            DevConst['NROFPTAT'] = 2
            
            self._package_num = 30
            self._package_size = 1401
            self._fs = 25
            self._NETD = 110
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '160x120.json'
            # Load calibration data from file
            self._load_calib_json(path)
            
        else:
            raise Exception('This Thermopile Array is not known.') 
         
        # Remaining DevConst can be derived
        DevConst['VDDaddr'] = \
            int(width*height+height/DevConst['NROFBLOCKS']*width)   
            
        DevConst['TAaddr']=DevConst['VDDaddr'] + 1
        DevConst['PTaddr']=DevConst['TAaddr'] + 1
            
        self._DevConst = DevConst        
        self._rowsPerBlock = int(height/DevConst['NROFBLOCKS'] / 2)
        self._pixelPerBlock = int(self._rowsPerBlock * width)
        self._PCSCALEVAL = 100000000
        
        # Derive order of serial data from DevConst
        # pixels
        pix = ['pix'+str(p) for p in range(0,width*height)]
        
        # electrical offsets
        no_e_off = int(height/DevConst['NROFBLOCKS'] * width)
        e_off = ['e_off'+str(e) for e in range(0,no_e_off)]
        
        # voltage
        vdd = ['Vdd'+str(v) for v in range(0,
                                    DevConst['TAaddr']-DevConst['VDDaddr'])]
        
        # ambient temperature
        T_amb = ['Tamb'+str(t) for t in range(0,
                                      DevConst['PTaddr']-DevConst['TAaddr'])]
        
        # PTAT
        no_ptat = int(DevConst['NROFBLOCKS']*DevConst['NROFPTAT'])
        PTAT = ['PTAT'+str(t) for t in range(0,no_ptat)]
        
        # ATC
        if DevConst['ATCaddr'] == 0:
            no_atc = 0
        else:
            no_atc = 2
            
        ATC = ['ATC'+str(a) for a in range(0,no_atc)]
        
        
        self._pix = pix
        self._e_off = e_off
        self._vdd = vdd
        self._T_amb = T_amb
        self._PTAT = PTAT
        self._ATC = ATC
        
        self._serial_data_order = pix + e_off + vdd + T_amb + PTAT + ATC
        
        
        # Calculate order of data in bcc file
        # initialize empty dictionary ee for eeprom adresses
        ee = {}
         
        # # Code by Christoph
        # ee['adr_pixcmin'] = np.array([0, 0])
        # ee['adr_pixcmax'] = np.array([0, 4])
        # ee['adr_gradScale'] = np.array([0, 8])
        # ee['adr_epsilon'] = np.array([0, 13])
        # ee['adr_vddMeas_th1'] = np.array([2, 6])
        # ee['adr_vddMeas_th2'] = np.array([2, 8])
        # ee['adr_ptatGrad'] = np.array([3, 4])
        # ee['adr_ptatOffset'] = np.array([3, 8])
        # ee['adr_ptat_th1'] = np.array([3, 12])
        # ee['adr_ptat_th2'] = np.array([3, 14])
        # ee['adr_vddScGrad'] = np.array([4, 14])
        # ee['adr_vddScOff'] = np.array([4, 15])
        # ee['adr_globalOff'] = np.array([5, 4])
        # ee['adr_globalGain'] = np.array([5, 5])
        
        # ee['adr_vddCompGrad'] = np.array([0, 0])
        # ee['adr_vddCompOff'] = np.array([0, 0])
        # ee['adr_thGrad'] = np.array([0, 0])
        # ee['adr_thOff'] = np.array([0, 0])
        # ee['adr_pij'] = np.array([0, 0])
        
        # if (width,height) == (32,32):
        #     ee['adr_vddCompGrad'][0] = 52
        #     ee['adr_vddCompOff'][0] = 84
        #     ee['adr_thGrad'][0] = 116
        #     ee['adr_thOff'][0] = 244
        #     ee['adr_pij'][0] = 372
            
        # elif (width,height) == (80,64):
        #     ee['adr_vddCompGrad'][0] = 128
        #     ee['adr_vddCompOff'][0] = 288
        #     ee['adr_thGrad'][0] = 448
        #     ee['adr_thOff'][0] = 768
        #     ee['adr_pij'][0] = 1408
            
        # elif (width,height) == (60,84):
        #     ee['adr_vddCompGrad'][0] = 293
        #     ee['adr_vddCompOff'][0] = 383
        #     ee['adr_thGrad'][0] = 473
        #     ee['adr_thOff'][0] = 788
        #     ee['adr_pij'][0] = 1418
            
        # elif (width,height) == (60,40):
            
        #     ee['adr_vddCompGrad'][0] = 1028
        #     ee['adr_vddCompOff'][0] = 1088
        #     ee['adr_thGrad'][0] = 1148
        #     ee['adr_thOff'][0] = 1448
        #     ee['adr_pij'][0] = 1748
        
        # elif (width,height) == (160,120):
            
        #     ee['adr_vddCompGrad'][0] = 528
        #     ee['adr_vddCompOff'][0] = 728
        #     ee['adr_thGrad'][0] = 928
        #     ee['adr_thOff'][0] = 3328
        #     ee['adr_pij'][0] = 5728
            
        # else:
        #     raise Exception('Implement EEPROM Map for this array type!')
            
        
        # self._eeprom_adresses = ee

    def _load_calib_json(self, path:Path):
        
        with open(path,'r') as file:
            eeprom_adresses = json.load(file)
        
        self._eeprom_adresses =  eeprom_adresses

            
    def _comp_eloff(self):
        
        pass
        
    
    def get_DevConst(self):
        return self._DevConst
    
    def get_serial_data_order(self):
        return self._serial_data_order
        
    def get_eeprom_adresses(self):
        return self._eeprom_adresses
    
    def set_LuT(self,LuT):
        self._LuT = LuT

    def set_BCC(self,bcc):
        self._bcc = bcc
    
    def import_LuT(self,lut_path):
        
        # Import Look up Table as pd.DataFrame
        LuT = pd.read_csv(lut_path, sep=',',header=0,index_col = 0)
        
        # Convert column header to int
        LuT.columns = np.array([int(c) for c in LuT.columns])
        
        # Load LuT from file
        # df_lut = pd.read_csv(lut_path.absolute(),header=None,
        #                      sep = '\t')
        
        # LuT = {}
        
        # LuT['ta'] = df_lut.loc[[0]].values[:,1:].astype(int).flatten()
        # LuT['digits'] = df_lut.loc[1::,0].values.astype(int).flatten()
        # LuT['to'] = df_lut.loc[1::].values[:,1::].astype(int)
        
        self._LuT = LuT
        
        return LuT    
        
    def import_BCC(self,bcc_path):
        """
        This is a copy of Read_BccData.py
        

        Parameters
        ----------
        bcc_path : pathlib.Path()
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Shorthand for EEPROM Adresses
        ee = self.get_eeprom_adresses()['EEPROM']
        
        # Initialize empty dict for return results
        bcc = {}
        
        ########################################
        # get all relevant data from .bcc file #
        ########################################
        
        # read hex data in bit by bit
        with open(bcc_path,'rb') as bcc_file:
            bcc_raw = bcc_file.read() 
        
        # Read and convert data according to provided json file
        for key in ee.keys():
            
            # Ge start and stop indices from addresses
            idx_start = int(ee[key]['adr_start'],0)
            idx_stop = int(ee[key]['adr_stop'],0)+1
            
            # Get raw value
            raw_val = bcc_raw[idx_start:idx_stop]
            
            # Convert raw value 
            bcc[key] = self._convert_raw_bcc(raw_val,ee[key]['dtype'])

        
        # Convert all EEPROM values from lists to numpy array
        for key in bcc.keys():
            bcc[key] = np.array(bcc[key]) 
        
        # Convert all arrays to appropriate shape and flip them
        # properly
        bcc['pij'] = np.array(bcc['pij']).reshape(self._npsize)
        bcc['thGrad'] = np.array(bcc['thGrad']).reshape(self._npsize)
        bcc['thOff'] = np.array(bcc['thOff']).reshape(self._npsize)
        
        NROFBLOCKS = self.get_DevConst()['NROFBLOCKS']
        vdd_size = (int(self._height/NROFBLOCKS),self._width)
        
        bcc['vddCompGrad'] = np.array(bcc['vddCompGrad']).reshape(vdd_size)
        bcc['vddCompOff'] = np.array(bcc['vddCompOff']).reshape(vdd_size)
        
        # The lower half needs to be flipped vertically
        bcc['pij'][int(self._height/2):,::] = \
            np.flipud(bcc['pij'][int(self._height/2):,::])
        
        bcc['thGrad'][int(self._height/2):,::] = \
            np.flipud(bcc['thGrad'][int(self._height/2):,::])
            
        bcc['thOff'][int(self._height/2):,::] = \
            np.flipud(bcc['thOff'][int(self._height/2):,::])

        
        bcc['vddCompGrad'][int(vdd_size[0]/2):,::] = \
            np.flipud(bcc['vddCompGrad'][int(vdd_size[0]/2):,::])
        
        bcc['vddCompOff'][int(vdd_size[0]/2):,::] = \
            np.flipud(bcc['vddCompOff'][int(vdd_size[0]/2):,::])
        
        
        self._bcc = bcc 
        
        return bcc
    

    def _convert_raw_bcc(self,raw_val:list,dtype:str):
        """
        Link to documentation of struct lybrary
        https://docs.python.org/3/library/struct.html#struct-format-strings
        """
        
        if dtype == 'float32':
            b_idx = np.arange(0,len(raw_val),4)
            conv_val = [struct.unpack('f',raw_val[b:b+4])[0] for b in  b_idx]
            
        elif dtype == 'uint8':            
            b_idx = np.arange(0,len(raw_val),1)
            conv_val = [struct.unpack('B',raw_val[b:b+1])[0] for b in  b_idx] 
            
        elif dtype == 'uint16':
            b_idx = np.arange(0,len(raw_val),2)
            conv_val = [struct.unpack('<H',raw_val[b:b+2])[0] for b in  b_idx] 

        elif dtype == 'int8':
            b_idx = np.arange(0,len(raw_val),1)
            conv_val = [struct.unpack('b',raw_val[b:b+1])[0] for b in  b_idx] 
            
        elif dtype == 'int16':
            b_idx = np.arange(0,len(raw_val),2)
            conv_val = [struct.unpack('<h',raw_val[b:b+2])[0] for b in  b_idx] 

        else:
            Exception('Unknown datatype')
            conv_val = None
        
        return conv_val
            
        
        

    def _comp_thermal_offset(self,df_meas:pd.Series):
        
        ''' Thermal offset compensation '''
        
        # Only for this function reverse self._size for easy use in numpy
        size = (self._size[1],self._size[0])
        
        Pixel = df_meas[self._pix] 
        
        
        # Get stuff for calculation
        ThGrad = self._bcc['thGrad'].reshape(size)
        avgPtat = df_meas[self._PTAT].mean().item()
        gradScale = self._bcc['gradScale']
        ThOffset = self._bcc['thOff'].reshape(size)
        
        V_th_comp = Pixel.values.reshape(size) -\
            (ThGrad*avgPtat) / np.power(2*np.ones(size),gradScale) -\
                ThOffset
         
        df_meas.loc[self._pix] = V_th_comp.flatten()
        
        return df_meas
    
    def _comp_electrical_offset(self,df_meas):
        
        ''' Electrical offset compensation '''
        ElOff = df_meas[self._e_off]
        
        # Only for this function reverse self._size for easy use in numpy
        size = (self._size[1],self._size[0])
        
        Pixel = df_meas[self._pix] 
        
        
        # Replicate electrical offsets corresponding to their pixels
        if self._DevConst['NROFPTAT']==2:
            ElOff_upper_half = ElOff.iloc[0:int(len(ElOff)/2)]
            ElOff_lower_half = ElOff.iloc[int(len(ElOff)/2)::]
            
            # Replicate the electrical offsets for the lower and upper
            # half NROFBLOCKS-times
            ElOff_upper_half = pd.concat([ElOff_upper_half]*\
                                         self._DevConst['NROFBLOCKS'],axis=0)
            ElOff_lower_half = pd.concat([ElOff_lower_half]*\
                                         self._DevConst['NROFBLOCKS'],axis=0)
            # Concatenate
            ElOff = pd.concat([ElOff_upper_half,
                               ElOff_lower_half])
            
        elif self._DevConst['NROFPTAT']==1:
            print('Yet to be implemented! Ask Bodo or Christoph!')
            return None
        
        V_el_comp = Pixel.values - ElOff.values
        
        df_meas.loc[self._pix] = V_el_comp
        
        return df_meas
        
    def _comp_vdd(self,df_meas):
        
        ''' Vdd compensation '''
        
        # Only for this function reverse self._size for easy use in numpy
        size = (self._size[1],self._size[0])
        
        Pixel = df_meas[self._pix] 
        
        # Get stuff for calculation
        vddCompGrad = self._bcc['vddCompGrad']
        vddCompOff = self._bcc['vddCompOff']
        vddScOff = self._bcc['vddScOff'].item()
        vddScGrad = self._bcc['vddScGrad'].item()


        vdd_av = df_meas[self._vdd].values.item()
        vdd_th1 = self._bcc['vddMeas_th1']
        vdd_th2 = self._bcc['vddMeas_th2']
        ptat_th1 = self._bcc['ptat_th1']
        ptat_th2 = self._bcc['ptat_th2']
        ptat_av = df_meas[self._PTAT].mean()

        # Replicate vddCompGrad and vddCompOff according to their
        # corresponding pixels
        
        # Replicate electrical offsets corresponding to their pixels
        if self._DevConst['NROFPTAT']==2:
            
            vdd_shape =  vddCompGrad.shape
            
            vddCompGrad_uh = vddCompGrad[0:int(vdd_shape[0]/2),:].flatten()
            vddCompGrad_lh = vddCompGrad[int(vdd_shape[0]/2):,:].flatten()
            
            vddCompOff_uh = vddCompOff[0:int(vdd_shape[0]/2),:].flatten()
            vddCompOff_lh = vddCompOff[int(vdd_shape[0]/2):,:].flatten()   
            
            # Replicate them all NROFBLOCKS-times
            vddCompGrad_uh = np.hstack([vddCompGrad_uh]*self._DevConst['NROFBLOCKS'])
            vddCompGrad_lh = np.hstack([vddCompGrad_lh]*self._DevConst['NROFBLOCKS'])
            vddCompOff_uh = np.hstack([vddCompOff_uh]*self._DevConst['NROFBLOCKS'])
            vddCompOff_lh = np.hstack([vddCompOff_lh]*self._DevConst['NROFBLOCKS'])
            
            # Concatenate
            vddCompGrad = np.hstack([vddCompGrad_uh,vddCompGrad_lh])
            vddCompOff = np.hstack([vddCompOff_uh,vddCompOff_lh])
                        
        elif self._DevConst['NROFPTAT']==1:
            print('Yet to be implemented! Ask Bodo or Christoph!')
            return None
        
        # Apply compensation 
        vdd = ((vddCompGrad*ptat_av)/(2**vddScGrad)+vddCompOff) / (2**vddScOff)
        vdd = vdd * (vdd_av - vdd_th1 - \
                     ((vdd_th2-vdd_th1)/(ptat_th2-ptat_th1))*(ptat_av-ptat_th1))
        
        V_vdd_comp = Pixel.values - vdd
        
        df_meas.loc[self._pix] = V_vdd_comp.flatten()
        
        return df_meas
    
    def _comp_sens(self,df_meas):
        
        # Only for this function reverse self._size for easy use in numpy
        size = (self._size[1],self._size[0])
        
        Pixel = df_meas[self._pix] 
        
        # Get stuff for calculation
        Pij = self._bcc['pij']
        PixCmin = self._bcc['pixcmin']
        PixCmax = self._bcc['pixcmax']
        GlobGain = self._bcc['globalGain']
        eps = self._bcc['epsilon']
        
        
        # Calculate Sensitivity coefficients
        PixC = (( Pij.reshape((-1,1)) * (PixCmax-PixCmin)  / 65535)  + PixCmin) \
            * eps/100 * GlobGain/10000
            
        # Compensate pixel voltage
        VijPixC =  Pixel * self._PCSCALEVAL / PixC.flatten()
        
        # Write to dataframe
        df_meas.loc[self._pix] = VijPixC
        
        return df_meas
        
    def _calc_Tamb0(self,df_meas):
        
        ptat_av = df_meas[self._PTAT].mean()
        
        ptat_grad = self._bcc['ptatGrad']
        ptat_off = self._bcc['ptatOffset']
        
        Tamb0 = ptat_av*ptat_grad+ptat_off
        
        df_meas.loc[self._T_amb] = Tamb0
        
        return df_meas
    
    def rawmeas_comp(self,df_meas):
        """
        Copy from Calc_CompTemp.py, no compensation of pixel sensitivity and
        no conversion in dK
        """
        
        df_calib = []
        
        for i in df_meas.index:
            
            df_frame = df_meas.loc[i]
            
            df_frame = self._comp_thermal_offset(df_frame.copy())
            df_frame = self._comp_electrical_offset(df_frame)
            df_frame = self._comp_vdd(df_frame)
            df_frame = self._comp_sens(df_frame)
            df_frame = self._calc_Tamb0(df_frame)
        
            # Convert back to DataFrame
            df_frame = pd.DataFrame(df_frame).transpose()
            df_calib.append(df_frame)
        
        df_calib = pd.concat(df_calib)
        
        return df_calib
    
    
    def rawmeas_to_dK(self,df_meas):
        """
        Copy from Calc_CompTemp.py, no compensation of pixel sensitivity and
        no conversion in dK
        """
        
        # Perform all compensation operations on data
        df_meas = self.rawmeas_comp(df_meas)
        
        
        # Load LuT
        LuT = self.LuT.copy()
                
        Warning('''rawmeas_to_dK returns compensated voltage, not dK! Remaining operations need to be implemented for dataframe format!''')
        
        # #############  step 5: multiply sensitivity coeff for each pixel #############
        # compPix[i][j] *= self._PCSCALEVAL / pixcScaled[i][j]
  
        # #############  step 6: find correct temp in lookup table and do a bilinear interpolation #############
        # for k in range(self._LuT['digits'].shape[0]):
        #     if(compPix[i][j] > (self._LuT['digits'][k])):
        #         to_row[i][j] = k
  
        # # calc ratios of neighboring entries within look-up-table
        # dto[0][i][j] = (self._LuT['digits'][to_row[i][j].astype(int) + 1] - compPix[i][j]) / (self._LuT['digits'][1] - self._LuT['digits'][0])
        # dto[1][i][j] = (compPix[i][j] - self._LuT['digits'][to_row[i][j].astype(int)]) / (self._LuT['digits'][1] - self._LuT['digits'][0])
  
        # # find all four surrounding entries within look-up-table
        # qto[0][i][j] = self._LuT['to'][to_row[i][j].astype(int)][ta_col]
        # qto[1][i][j] = self._LuT['to'][to_row[i][j].astype(int) + 1][ta_col]
        # qto[2][i][j] = self._LuT['to'][to_row[i][j].astype(int)][ta_col + 1]
        # qto[3][i][j] = self._LuT['to'][to_row[i][j].astype(int) + 1][ta_col + 1]
  
        # # interpolate along ambient temperature
        # rto[0][i][j] = dta[0] * qto[0][i][j] + dta[1] * qto[2][i][j]
        # rto[1][i][j] = dta[0] * qto[1][i][j] + dta[1] * qto[3][i][j]
  
        # # interpolate along measured digits
        # compPix[i][j] = rto[0][i][j] * dto[0][i][j] + rto[1][i][j] * dto[1][i][j]
  
        # #############  step 7: add GlobalOffset #############
        # compPix[i][j] += self._bcc['globalOff']

        
        return df_meas
    
    def _binary_mask(self,r_lim:float)->np.ndarray:
        """
        Creates a binary mask that is 1 if distance from image center
        is less of equal to r and 0 otherwise

        Parameters
        ----------
        r_lim : float
            Maximal distance in pixels from the center, where mask is supposed 
            to be 1.

        Returns
        -------
        None.

        """
        
        # Calculate center
        x_center = (self._width-1) / 2
        y_center = (self._height-1) / 2
        
        # Initialize mask
        mask = np.zeros(( self._height, self._width))
        
        # Create meshgrid of coordinates
        y, x = np.meshgrid(np.arange(self._height), np.arange(self._width),
                           indexing = 'ij')

        
        # Calculate the distance of each point from the center
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        
        # Use numpy.where to set values to 0 if d is larger than r_lim
        mask = np.where(r > r_lim, 0, 1)
        
        return mask
        