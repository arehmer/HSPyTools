# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:16:48 2023

@author: Rehmer
"""
import numpy as np
import ctypes
import pandas as pd

class TPArray():
    """
    Class contains hard-coded properties of Thermopile-Arrays relevant
    for reading from Bytestream
    """
    
    def __init__(self,width,height):
        
        # Calculate order of data in bds file
        DevConst = {}
        
        if (width,height) == (8,8):
            DevConst['VDDaddr']=128
            DevConst['TAaddr']=129
            DevConst['PTaddr']=130
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=1
            DevConst['NROFPTAT']=1
            self._fs = 160
            self._NETD = 100
            
        elif (width,height) == (16,16):
            DevConst['VDDaddr']=384
            DevConst['TAaddr']=385
            DevConst['PTaddr']=386
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=2
            DevConst['NROFPTAT']=2
            self._fs = 70
            self._NETD = 130
        
        elif (width,height) == (32,32):
            DevConst['VDDaddr']=1280
            DevConst['TAaddr']=1281
            DevConst['PTaddr']=1282
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=4
            DevConst['NROFPTAT']=2
            self._fs = 27
            self._NETD = 140
            
        elif (width,height) == (80,64):
            DevConst['VDDaddr']=6400
            DevConst['TAaddr']=6401
            DevConst['PTaddr']=6402
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=4
            DevConst['NROFPTAT']=2
            self._fs = 41
            self._NETD = 90 
            self._NETD = 70
                      
        elif (width,height) == (120,84):
            DevConst['VDDaddr']=11760
            DevConst['TAaddr']=11761
            DevConst['PTaddr']=11762
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=6
            DevConst['NROFPTAT']=2
            self._fs = 28
            self._NETD = 130
            
        elif (width,height) == (60,40):
            DevConst['VDDaddr']=2880
            DevConst['TAaddr']=2881
            DevConst['PTaddr']=2882
            DevConst['ATCaddr']=2892
            DevConst['NROFBLOCKS']=5
            DevConst['NROFPTAT']=2
            self._fs = 47

        elif (width,height) == (60,84):
            DevConst['VDDaddr']=5760
            DevConst['TAaddr']=5761
            DevConst['PTaddr']=5762
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=7
            DevConst['NROFPTAT']=2
            self._fs = 25
            self._NETD = 110
            
            
        self._DevConst = DevConst
        self._width = width
        self._height = height
        self._size = (width,height)
        self._npsize = (height,width)
        
        self._rowsPerBlock = int(height/DevConst['NROFBLOCKS'] / 2)
        self._pixelPerBlock = int(self._rowsPerBlock * width)
        self._PCSCALEVAL = 100000000
        
        # Order of serial data from sensor type
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
         
        # Code by Christoph
        ee['adr_pixcmin'] = np.array([0, 0])
        ee['adr_pixcmax'] = np.array([0, 4])
        ee['adr_gradScale'] = np.array([0, 8])
        ee['adr_epsilon'] = np.array([0, 13])
        ee['adr_vddMeas_th1'] = np.array([2, 6])
        ee['adr_vddMeas_th2'] = np.array([2, 8])
        ee['adr_ptatGrad'] = np.array([3, 4])
        ee['adr_ptatOffset'] = np.array([3, 8])
        ee['adr_ptat_th1'] = np.array([3, 12])
        ee['adr_ptat_th2'] = np.array([3, 14])
        ee['adr_vddScaling'] = np.array([4, 14])
        ee['adr_vddScalingOff'] = np.array([4, 15])
        ee['adr_globalOff'] = np.array([5, 4])
        ee['adr_globalGain'] = np.array([5, 5])
        
        ee['adr_vddCompGrad'] = np.array([0, 0])
        ee['adr_vddCompOff'] = np.array([0, 0])
        ee['adr_thGrad'] = np.array([0, 0])
        ee['adr_thOff'] = np.array([0, 0])
        ee['adr_pij'] = np.array([0, 0])
        
        if (width,height) == (32,32):
            ee['adr_vddCompGrad'][0] = 52
            ee['adr_vddCompOff'][0] = 84
            ee['adr_thGrad'][0] = 116
            ee['adr_thOff'][0] = 244
            ee['adr_pij'][0] = 372
            
        elif (width,height) == (80,64):
            ee['adr_vddCompGrad'][0] = 128
            ee['adr_vddCompOff'][0] = 288
            ee['adr_thGrad'][0] = 448
            ee['adr_thOff'][0] = 768
            ee['adr_pij'][0] = 1408
            
        elif (width,height) == (60,84):
            ee['adr_vddCompGrad'][0] = 293
            ee['adr_vddCompOff'][0] = 383
            ee['adr_thGrad'][0] = 473
            ee['adr_thOff'][0] = 788
            ee['adr_pij'][0] = 1418
        
        self._eeprom_adresses = ee
        
                
    def _comp_eloff():
        
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
        
        # Load LuT from file
        df_lut = pd.read_csv(lut_path.absolute(),header=None,
                             sep = '\t')
        
        LuT = {}
        
        LuT['ta'] = df_lut.loc[[0]].values[:,1:].astype(int).flatten()
        LuT['digits'] = df_lut.loc[1::,0].values.astype(int).flatten()
        LuT['to'] = df_lut.loc[1::].values[:,1::].astype(int)
        
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
        ee = self.get_eeprom_adresses()
        
        # Initialize empty dict for return results
        bcc = {}
        
        ########################################
        # get all relevant data from .bcc file #
        ########################################
        
        # read hex data as char
        bccData = np.fromfile(bcc_path.absolute(), dtype='uint8')
        
        # reshape to sort EEPROM content as rows of 16 addresses
        if(self._size[0] == 32) and (self._size[1] == 32):
            bccData = np.reshape(bccData, (512, 16))
        elif (self._size[0] == 60) and (self._size[1] == 84):
            bccData = np.reshape(bccData, (2048, 16))
        
        ##########################################################
        # calculate all numbers from individual EEPROM addresses #
        ##########################################################
        
        # calculate PixCmin (saved as float in EEPROM)
        pixcmin_el = np.array([[bccData[ee['adr_pixcmin'][0], ee['adr_pixcmin'][1]]],
                               [bccData[ee['adr_pixcmin'][0], ee['adr_pixcmin'][1] + 1]],
                               [bccData[ee['adr_pixcmin'][0], ee['adr_pixcmin'][1] + 2]],
                               [bccData[ee['adr_pixcmin'][0], ee['adr_pixcmin'][1] + 3]]])
        if pixcmin_el[3] > 127:
            pixcmin_sign = -1
        else:
            pixcmin_sign = 1
        pixcmin_exp = np.floor(((((pixcmin_el[3] * 256) + pixcmin_el[2]) % 32768) / 128) - 127).astype(int)[0]
        pixcmin_mantissa = ((pixcmin_el[2] * 65536 + pixcmin_el[1] * 256 + pixcmin_el[0]) % (16777216 / 2)).astype(int)[0]
        pixcmin = pixcmin_sign * (2.0 ** pixcmin_exp) * ((pixcmin_mantissa + 2 ** 23) / (2 ** 23))
        
        # calculate PixCmax (saved as float in EEPROM)
        pixcmax_el = np.array([[bccData[ee['adr_pixcmax'][0], ee['adr_pixcmax'][1]]],
                               [bccData[ee['adr_pixcmax'][0], ee['adr_pixcmax'][1] + 1]],
                               [bccData[ee['adr_pixcmax'][0], ee['adr_pixcmax'][1] + 2]],
                               [bccData[ee['adr_pixcmax'][0], ee['adr_pixcmax'][1] + 3]]])
        if pixcmax_el[3] > 127:
            pixcmax_sign = -1
        else:
            pixcmax_sign = 1
        pixcmax_exp = np.floor(((((pixcmax_el[3] * 256) + pixcmax_el[2]) % 32768) / 128) - 127).astype(int)[0]
        pixcmax_mantissa = ((pixcmax_el[2] * 65536 + pixcmax_el[1] * 256 + pixcmax_el[0]) % (16777216 / 2)).astype(int)[0]
        pixcmax = pixcmax_sign * (2.0 ** pixcmax_exp) * ((pixcmax_mantissa + 2 ** 23) / (2 ** 23))
        
        # get gradscale
        gradScale = bccData[ee['adr_gradScale'][0], ee['adr_gradScale'][1]]
        
        # get epsilon
        epsilon = bccData[ee['adr_epsilon'][0], ee['adr_epsilon'][1]]
        
        # get Vdd Meas Th1
        vddMeas_th1 = bccData[ee['adr_vddMeas_th1'][0], ee['adr_vddMeas_th1'][1] + 1] * 256 + bccData[ee['adr_vddMeas_th1'][0], ee['adr_vddMeas_th1'][1]]
        
        # get Vdd Meas Th2
        vddMeas_th2 = bccData[ee['adr_vddMeas_th2'][0], ee['adr_vddMeas_th2'][1] + 1] * 256 + bccData[ee['adr_vddMeas_th2'][0], ee['adr_vddMeas_th2'][1]]
        
        # calculate PTAT-Gradient (saved as float in EEPROM)
        ptatGrad_el = np.array([[bccData[ee['adr_ptatGrad'][0], ee['adr_ptatGrad'][1]]],
                                [bccData[ee['adr_ptatGrad'][0], ee['adr_ptatGrad'][1] + 1]],
                                [bccData[ee['adr_ptatGrad'][0], ee['adr_ptatGrad'][1] + 2]],
                                [bccData[ee['adr_ptatGrad'][0], ee['adr_ptatGrad'][1] + 3]]])
        if ptatGrad_el[3] > 127:
            ptatGrad_sign = -1
        else:
            ptatGrad_sign = 1
        ptatGrad_exp = np.floor(((((ptatGrad_el[3] * 256) + ptatGrad_el[2]) % 32768) / 128) - 127).astype(int)[0]
        ptatGrad_mantissa = ((ptatGrad_el[2] * 65536 + ptatGrad_el[1] * 256 + ptatGrad_el[0]) % (16777216 / 2)).astype(int)[0]
        ptatGrad = ptatGrad_sign * (2.0 ** ptatGrad_exp) * ((ptatGrad_mantissa + 2 ** 23) / (2 ** 23))
        
        # calculate PTAT-Offset (saved as float in EEPROM)
        ptatOffset_el = np.array([[bccData[ee['adr_ptatOffset'][0], ee['adr_ptatOffset'][1]]],
                                [bccData[ee['adr_ptatOffset'][0], ee['adr_ptatOffset'][1] + 1]],
                                [bccData[ee['adr_ptatOffset'][0], ee['adr_ptatOffset'][1] + 2]],
                                [bccData[ee['adr_ptatOffset'][0], ee['adr_ptatOffset'][1] + 3]]])
        if ptatOffset_el[3] > 127:
            ptatOffset_sign = -1
        else:
            ptatOffset_sign = 1
        ptatOffset_exp = np.floor(((((ptatOffset_el[3] * 256) + ptatOffset_el[2]) % 32768) / 128) - 127).astype(int)[0]
        ptatOffset_mantissa = ((ptatOffset_el[2] * 65536 + ptatOffset_el[1] * 256 + ptatOffset_el[0]) % (16777216 / 2)).astype(int)[0]
        ptatOffset = ptatOffset_sign * (2.0 ** ptatOffset_exp) * ((ptatOffset_mantissa + 2 ** 23) / (2 ** 23))
        
        # get PTAT(Th1)
        ptat_th1 = bccData[ee['adr_ptat_th1'][0], ee['adr_ptat_th1'][1] + 1] * 256 + bccData[ee['adr_ptat_th1'][0], ee['adr_ptat_th1'][1]]
        
        # get PTAT(Th2)
        ptat_th2 = bccData[ee['adr_ptat_th2'][0], ee['adr_ptat_th2'][1] + 1] * 256 + bccData[ee['adr_ptat_th2'][0], ee['adr_ptat_th2'][1]]
        
        # get VddScaling
        vddScaling = bccData[ee['adr_vddScaling'][0], ee['adr_vddScaling'][1]]
        
        # get VddScalingOff
        vddScalingOff = bccData[ee['adr_vddScalingOff'][0], ee['adr_vddScalingOff'][1]]
        
        # get GlobalOff
        globalOff = ctypes.c_int8(bccData[ee['adr_globalOff'][0], ee['adr_globalOff'][1]]).value
        
        # get GlobalGain
        globalGain = bccData[ee['adr_globalGain'][0], ee['adr_globalGain'][1] + 1] * 256 + bccData[ee['adr_globalGain'][0], ee['adr_globalGain'][1]]
        
        # get VddCompGrad and VddCompOffset
        vddCompGrad_el = np.zeros([self._pixelPerBlock * 2])
        vddCompOff_el = np.zeros([self._pixelPerBlock * 2])
        for i in range(int(self._pixelPerBlock * 2 / 8)):
            for j in range(8):
                vddCompGrad_el[8 * i + j] = ctypes.c_int16(bccData[ee['adr_vddCompGrad'][0] + i, ee['adr_vddCompGrad'][1] + 2 * j + 1] * 256 + bccData[ee['adr_vddCompGrad'][0] + i, ee['adr_vddCompGrad'][1] + 2 * j]).value
                vddCompOff_el[8 * i + j] = ctypes.c_int16(bccData[ee['adr_vddCompOff'][0] + i, ee['adr_vddCompOff'][1] + 2 * j + 1] * 256 + bccData[ee['adr_vddCompOff'][0] + i, ee['adr_vddCompOff'][1] + 2 * j]).value
        # sort VddCompGrad and VddCompOffset
        vddCompGrad = np.zeros([self._rowsPerBlock * 2, self._size[0]])
        vddCompOff = np.zeros([self._rowsPerBlock * 2, self._size[0]])
        m = 0
        n = 0
            # top half
        for i in range(self._pixelPerBlock):
            vddCompGrad[m][n] = vddCompGrad_el[i]
            vddCompOff[m][n] = vddCompOff_el[i]
            n += 1
            if (n == self._size[0]):
                n = 0
                m += 1
        
            # bottom half
        m = self._rowsPerBlock * 2 - 1
        n = 0
        for i in range(self._pixelPerBlock):
            vddCompGrad[m][n] = vddCompGrad_el[self._pixelPerBlock + i]
            vddCompOff[m][n] = vddCompOff_el[self._pixelPerBlock + i]
            n += 1
            if (n == self._size[0]):
                n = 0
                m -= 1
        
        # get ThGrad and ThOffset
        thGrad_el = np.zeros([self._size[0] * self._size[1]])
        thOff_el = np.zeros([self._size[0] * self._size[1]])
        # 32x32
        if(self._size[0] == 32) and (self._size[1] == 32):
            for i in range(int(self._size[0] * self._size[1] / 8)):
                for j in range(8):
                    thGrad_el[j + 8 * i] = ctypes.c_int16((bccData[ee['adr_thGrad'][0] + i, ee['adr_thGrad'][1] + 2 * j + 1] * 256 + bccData[ee['adr_thGrad'][0] + i, ee['adr_thGrad'][1] + 2 * j])).value
                    thOff_el[j + 8 * i] = ctypes.c_int16(bccData[ee['adr_thOff'][0] + i, ee['adr_thOff'][1] + 2 * j + 1] * 256 + bccData[ee['adr_thOff'][0] + i, ee['adr_thOff'][1] + 2 * j]).value
        # 80x64
        elif(self._size[0] == 80) and (self._size[1] == 64):
            for i in range(int(self._size[0] * self._size[1] / 16)):
                for j in range(16):
                    thGrad_el[j + 16 * i] = ctypes.c_int8(bccData[ee['adr_thGrad'][0] + i, ee['adr_thGrad'][1] + j]).value
            for i in range(int(self._size[0] * self._size[1] / 8)):
                for j in range(8):
                    thOff_el[j + 8 * i] = ctypes.c_int16(bccData[ee['adr_thOff'][0] + i, ee['adr_thOff'][1] + 2 * j + 1] * 256 + bccData[ee['adr_thOff'][0] + i, ee['adr_thOff'][1] + 2 * j]).value
        elif(self._size[0] == 60) and (self._size[1] == 84):
            for i in range(int(self._size[0] * self._size[1] / 16)):
                for j in range(16):
                    thGrad_el[j + 16 * i] = ctypes.c_int8(bccData[ee['adr_thGrad'][0] + i, ee['adr_thGrad'][1] + j]).value
            for i in range(int(self._size[0] * self._size[1] / 8)):
                for j in range(8):
                    thOff_el[j + 8 * i] = ctypes.c_int16(bccData[ee['adr_thOff'][0] + i, ee['adr_thOff'][1] + 2 * j + 1] * 256 + bccData[ee['adr_thOff'][0] + i, ee['adr_thOff'][1] + 2 * j]).value
        
        # sort ThGrad and ThOffset
        thGrad = np.zeros([self._size[1], self._size[0]])
        thOff = np.zeros([self._size[1], self._size[0]])
        m = 0
        n = 0
            # top half
        for i in range((int)(self._size[0] * self._size[1] / 2)):
            thGrad[m][n] = thGrad_el[i]
            thOff[m][n] = thOff_el[i]
            n += 1
            if (n == self._size[0]):
                n = 0
                m += 1
        
            # bottom half
        m = self._size[1] - 1
        n = 0
        i = 0
        for i in range((int)(self._size[0] * self._size[1] / 2)):
            thGrad[m][n] = thGrad_el[(int)(self._size[0] * self._size[1] / 2) + i]
            thOff[m][n] = thOff_el[(int)(self._size[0] * self._size[1] / 2) + i]
            n += 1
            if (n == self._size[0]):
                n = 0
                m -= 1
        
        # get Pij
        pij_el = np.zeros([self._size[0] * self._size[1]])
        for i in range(int(self._size[0] * self._size[1] / 8)):
            for j in range(8):
                pij_el[j + 8 * i] = bccData[ee['adr_pij'][0] + i, ee['adr_pij'][1] + 2 * j + 1] * 256 + bccData[ee['adr_pij'][0] + i, ee['adr_pij'][1] + 2 * j]
        
        # sort Pij
        pij = np.zeros([self._size[1], self._size[0]])
        m = 0
        n = 0
            # top half
        for i in range((int)(self._size[0] * self._size[1] / 2)):
            pij[m][n] = pij_el[i]
            n += 1
            if (n == self._size[0]):
                n = 0
                m += 1
        
            # bottom half
        m = self._size[1] - 1
        n = 0
        i = 0
        for i in range((int)(self._size[0] * self._size[1] / 2)):
            pij[m][n] = pij_el[(int)(self._size[0] * self._size[1] / 2) + i]
            n += 1
            if (n == self._size[0]):
                n = 0
                m -= 1

        # In the end package all the important stuff in a dict and return
        
        bcc['gradScale'] = gradScale
        bcc['vddScaling'] = vddScaling
        bcc['vddScalingOff'] = vddScalingOff
        bcc['pij'] = pij
        bcc['pixcmin'] = pixcmin
        bcc['pixcmax'] = pixcmax
        bcc['epsilon'] = epsilon
        bcc['globalGain'] = globalGain
        bcc['ptatGrad'] = ptatGrad
        bcc['ptatOffset'] = ptatOffset
        bcc['thGrad'] = thGrad
        bcc['thOff'] = thOff
        bcc['vddCompGrad'] = vddCompGrad
        bcc['vddCompOff'] = vddCompOff
        bcc['vddMeas_th1'] = vddMeas_th1
        bcc['vddMeas_th2'] = vddMeas_th2
        bcc['ptat_th1'] = ptat_th1
        bcc['ptat_th2'] = ptat_th2
        bcc['globalOff'] = globalOff
        
        self._bcc = bcc 
        
        return bcc
    


    def _comp_thermal_offset(self,df_meas):
        
        ''' Thermal offset compensation '''
        
        # Only for this function reverse self._size for easy use in numpy
        size = (self._size[1],self._size[0])
        
        Pixel = df_meas[self._pix] 
        
        # Get stuff for calculation
        ThGrad = self._bcc['thGrad'].reshape(size)
        avgPtat = df_meas[self._PTAT].mean()
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
        vddScOff = self._bcc['vddScalingOff'].item()
        vddScGrad = self._bcc['gradScale'].item()
        vddScale = self._bcc['vddScaling'].item()

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
                     (vdd_th2-vdd_th1)/(ptat_th2-ptat_th1)*(ptat_av-ptat_th1))
        
        

        V_vdd_comp = Pixel.values - vdd
        
        df_meas.loc[self._pix] = V_vdd_comp.flatten()
        
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
        
        df_meas = self._comp_thermal_offset(df_meas)
        df_meas = self._comp_electrical_offset(df_meas)
        df_meas = self._comp_vdd(df_meas)
        df_meas = self._calc_Tamb0(df_meas)
        
        return df_meas
    
    
    def rawmeas_to_dK(self,df_meas):
        """
        Copy from Calc_CompTemp.py, no compensation of pixel sensitivity and
        no conversion in dK
        """
        
        df_meas = self.rawmeas_comp(df_meas)
        
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