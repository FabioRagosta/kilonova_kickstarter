import numpy as np
import pandas as pd
import os, sys, time, glob
import lsst.sims.maf.metrics as metrics
from lsst.sims.utils import uniformSphere
import lsst.sims.maf.slicers as slicers
from lsst.sims.photUtils import Dust_values
from lsst.sims.photUtils import Bandpass, SignalToNoise, PhotometricParameters, calcMagError_m5, calcGamma

class GRBKN_obs(metrics.BaseMetric):
    def __init__(self, metricName='GRBKN_obs', mjdCol='observationStartMJD', 
                 RACol='fieldRA', DecCol='fieldDec',filterCol='filter', m5Col='fiveSigmaDepth', 
                 exptimeCol='visitExposureTime',nightCol='night',vistimeCol='visitTime', snrlim=5, ptsNeeded=2, mjd0=59853.5,
                 data_path='./lc', obs_path='./obs_lc',surveyduration=10,Filter_selection = False,**kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.nightCol = nightCol
        self.vistimeCol = vistimeCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.data_path = data_path
        self.obs_path = obs_path
        self.surveyduration=surveyduration
        self.mjd0=mjd0
        self.Filter_selection=Filter_selection
        self.snrlim = snrlim
        if not os.path.exists(self.obs_path):
            os.mkdir('./obs_lc')
        self.bandpass = Bandpass(wavelen=np.array([480.2,623.1,754.2]),
                                 sb=np.array([0.1,0.1,0.1]),wavelen_min=380, wavelen_max=850, wavelen_step=10)
        super(GRBKN_obs, self).__init__(col=[self.mjdCol,self.m5Col, self.filterCol,self.RACol,
                                                                   self.DecCol,self.exptimeCol,self.nightCol,
                                                                   self.vistimeCol],
                                                       metricDtype='object', units='',
                                                       metricName=metricName, **kwargs)
        self.bandpass = Bandpass(wavelen=np.array([480.2,623.1,754.2]),sb=np.array([0.1,0.1,0.1]),wavelen_min=380, wavelen_max=850, wavelen_step=10)
        self.photparam = PhotometricParameters() 
    def coadd(self, data):
        """
        Method to coadd data per band and per night
        Parameters
        ------------
        data : `pd.DataFrame`
            pandas df of observations
        Returns
        -------
        coadded data : `pd.DataFrame`
        """

        keygroup = [self.filterCol, self.nightCol]

        data.sort_values(by=keygroup, ascending=[
                         True, True], inplace=True)

        coadd_df = data.groupby(keygroup).agg({self.vistimeCol: ['sum'],
                                               self.exptimeCol: ['sum'],
                                               self.mjdCol: ['mean'],
                                               self.RACol: ['mean'],
                                               self.DecCol: ['mean'],
                                               self.m5Col: ['mean']}).reset_index()

        coadd_df.columns = [self.filterCol, self.nightCol, 
                            self.vistimeCol, self.exptimeCol, self.mjdCol,
                            self.RACol, self.DecCol, self.m5Col]

        coadd_df.loc[:, self.m5Col] += 1.25 * \
            np.log10(coadd_df[self.vistimeCol]/30.)

        coadd_df.sort_values(by=[self.filterCol, self.nightCol], ascending=[
                             True, True], inplace=True)

        return coadd_df.to_records(index=False)
    def read_lightCurve(self, file):
        """Reads in a csv file, from the simulated ligh curves, time and mag columns for each filter
        Returns
        -------
        numpy.ndarray
            The data read from the ascii text file, in a numpy structured array with columns
            'ph' (phase / epoch, in days), 'mag' (magnitude), 'flt' (filter for the magnitude).
        """
        self.lcv_template = pd.read_csv(file)
        self.transDuration = self.lcv_template['time'].max() - self.lcv_template['time'].min()
    def make_lightCurve(self, time, filters):
        """Turn lightcurve definition into magnitudes at a series of times.
        Parameters
        ----------
        time : numpy.ndarray
            The times of the observations.
        filters : numpy.ndarray
            The filters of the observations.
        Returns
        -------
        numpy.ndarray
             The magnitudes of the transient at the times and in the filters of the observations.
        """
        flt = np.unique(filters)
        lcMags = np.zeros(time.size, dtype=float)
        for key in set(flt):
            # Interpolate the lightcurve template to the times of the observations, in this filter.
            temp_ph=np.array(self.lcv_template['time'], float)
            lcMags[filters==key] = np.interp(time[filters==key], temp_ph,
                                        np.array(self.lcv_template[key], float))
        return lcMags
    
    def run(self,dataSlice, slicePoint=None):
        # Sort the entire dataSlice in order of time.
        dataSlice.sort(order=self.mjdCol)
        dataSlice = self.coadd(pd.DataFrame(dataSlice))
        bandpass = self.bandpass
        photparam = self.photparam
        calcSNR_m5=np.vectorize(SignalToNoise.calcSNR_m5)
        
        obs_filter = dataSlice[self.filterCol]
        obs = dataSlice[self.mjdCol]          
        obs_m5 = dataSlice[self.m5Col]
        
        lcfile = glob.glob(self.data_path+'/*.csv')
        for file in lcfile:
            print(file)
            lcname = file.split('/')[-1]
            self.read_lightCurve(file)
            
            lcNumberStart = -1 * np.floor((dataSlice[self.mjdCol].min() - self.mjd0) / self.transDuration)
            # Calculate the time/epoch for each lightcurve.
            lcEpoch = (obs-self.mjd0) % self.transDuration
            # Identify the observations which belong to each distinct light curve.
            lcNumber = np.floor((obs-self.mjd0 )  / self.transDuration) + lcNumberStart
            lcNumberStart = lcNumber.max()
            ulcNumber = np.unique(lcNumber)
            lcLeft = np.searchsorted(lcNumber, ulcNumber, side='left')
            lcRight = np.searchsorted(lcNumber, ulcNumber, side='right')
            lcMags = self.make_lightCurve(lcEpoch, obs_filter) 
            lcSNR,_ = calcSNR_m5(lcMags, bandpass, obs_m5, photparam)
            lcpoints_AboveThresh = np.zeros(np.shape(lcSNR), dtype=bool) 

            nobj = np.array([0,0],[('detected','i4'),('undetected','i4')])            
            for f in np.unique(obs_filter):                    
                filtermatch = np.where(obs_filter == f)
                lcpoints_AboveThresh[filtermatch] = np.where(lcMags[filtermatch] <= obs_m5[filtermatch],True,lcpoints_AboveThresh[filtermatch])

            for j,(lcN, le, ri) in enumerate(zip(ulcNumber, lcLeft, lcRight)):
                if le == ri:
                    # Skip the rest of this loop, go on to the next lightcurve.
                    continue
                Dpoints = np.sum(lcpoints_AboveThresh[le:ri], axis= 0) #counts the number of detected points
                if self.Filter_selection:
                    nfilt_det = np.zeros(np.shape(lcSNR)[0], dtype=bool)
                    for f in self.nFilters:                    
                        filtermatch = np.where(obs_filter == f)  
                        Dpoints = np.sum(lcpoints_AboveThresh[filtermatch],axis=0)
                        nfilt_det= np.where(Dpoints>=self.ptsNeeded,True,nfilt_det)
                    nobj['detected']+=np.sum(nfilt_det)
                    nobj['undetected']+= np.size(nfilt_det)-np.sum(nfilt_det)

                else:
                    nobj['detected']+=np.sum(np.where(Dpoints>self.ptsNeeded))
                    nobj['undetected']+=np.sum(np.where(Dpoints<self.ptsNeeded))

                # producing a file of LSST observed lightcurves if le != ri:
                mag = lcMags[le:ri][lcpoints_AboveThresh[le:ri]]
                filters = obs_filter[le:ri][lcpoints_AboveThresh[le:ri]]
                epochs = obs[le:ri][lcpoints_AboveThresh[le:ri]]
                snr = lcSNR[le:ri][lcpoints_AboveThresh[le:ri]]
                gamma = np.array([calcGamma(self.bandpass, m, self.photparam) 
                                  for m in obs_m5[le:ri][lcpoints_AboveThresh[le:ri]]])
                merr, _= calcMagError_m5(mag, bandpass, obs_m5[le:ri][lcpoints_AboveThresh[le:ri]], 
                                            photparam, gamma=gamma)

                if np.size(mag)>self.ptsNeeded:
                    total_lc = pd.DataFrame(np.array([epochs[snr>self.snrlim],
                                            filters[snr>self.snrlim],
                                            mag[snr>self.snrlim],
                                            merr[snr>self.snrlim],
                                            ]).T, columns=['time','filter', 'mag','merr']).to_csv(self.obs_path+'/obs_{}_'.format(j)+
                                                                                                              lcname)
        return nobj
