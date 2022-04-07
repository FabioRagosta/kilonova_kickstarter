import numpy as np
import pandas as pd
import os, sys, time, glob
import lsst.sims.maf.metrics as metrics
from lsst.sims.utils import uniformSphere
import lsst.sims.maf.slicers as slicers
from lsst.sims.photUtils import Dust_values
from lsst.sims.photUtils import Bandpass, SignalToNoise, PhotometricParameters, calcMagError_m5, calcGamma

class GRBKN_obs(metrics.BaseMetric):
    def __init__(self, metricName='KNePopMetric', mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', ptsNeeded=2, file_list=None, mjd0=59853.5,
                 data_path='./lc', obs_path='./obs_lc',**kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.data_path = data_path
        self.obs_path = obs_path
        if not os.path.exist(self.obs_path):
            os.mkdir('./obs_lc')
        self.bandpass = Bandpass(wavelen=np.array([480.2,623.1,754.2]),
                                 sb=np.array([0.1,0.1,0.1]),wavelen_min=380, wavelen_max=850, wavelen_step=10)
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
        if not os.path.isfile(asciifile):
            raise IOError('Could not find lightcurve file %s' % (file))
        self.lcv_template = pd.read_csv(file)
        
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
        lcMags = pd.DataFrame(columns = flt)
        for key in set(flt):
            # Interpolate the lightcurve template to the times of the observations, in this filter.
            temp_ph=np.array(self.lcv_template['time'], float)
            lcMags[key] = np.interp(time, temp_ph,
                                        np.array(self.lcv_template[key], float))
        return lcMags
    
    def run(self,dataSlice, slicePoint=None):
        # Sort the entire dataSlice in order of time.
        dataSlice.sort(order=self.mjdCol)
        dataSlice = self.coadd(pd.DataFrame(dataSlice))
        bandpass = self.bandpass
        photparam = self.photparam
        calcSNR_m5=np.vectorize(SignalToNoise.calcSNR_m5)
        
        lcfile = glob.glob(self.data_path+'/*.csv')
        for file in lcfile:
            lcname = file.split('/')[-1]
            self.read_lightCurve(file)
            lcMags_ = self.make_lightCurve(lcEpoch, obs_filter[indexlc])
            #broadcast the lc in the proper format for the MAF
            lcMags = np.array([np.concatenate(lcMags_.values), 
                               np.concatenate([np.repeat(k, np.size(lcMags_[k])) for k in lcMags_.columns])],
                 dtype=[('mag', 'f4'), ('flt', 'utf-8')])
            lcSNR,_ = calcSNR_m5(lcMags, bandpass, obs_m5[indexlc], photparam)
            lcpoints_AboveThresh = np.zeros(len(lcSNR), dtype=bool) 

            nobj = np.array([0,0],[('detected','i4'),('undetected','i4')])            
            for f in np.unique(obs_filter[indexlc]):                    
                filtermatch = np.where(obs_filter[indexlc] == f)
                lcpoints_AboveThresh[filtermatch] = np.where(lcMags_temp[filtermatch] <= obs_m5[indexlc][filtermatch],True,lcpoints_AboveThresh[filtermatch])
            Dpoints = np.sum(lcpoints_AboveThresh) #counts the number of detected points
            if self.Filter:
                nfilt_det = []
                for f in self.nFilters:                    
                    filtermatch = np.where(obs_filter[indexlc] == f)                               
                    if Dpoints[filtermatch]>=self.ptsNeeded: nfilt_det.append(True)

                if any(nfilt_det): 
                    nobj['detected']+=1
                else:
                    nobj['undetected']+=1

            else:
                if Dpoints>=self.ptsNeeded: 
                    nobj['detected']+=1
                else:
                    nobj['undetected']+=1

                # producing a file of LSST observed lightcurves
                mag = lcMags['mag'][lcpoints_AboveThresh]
                filters = lcMags['flt'][lcpoints_AboveThresh]
                epochs = obs[indexlc][lcpoints_AboveThresh][filtermatch]
                snr = lcSNR[lcpoints_AboveThresh][filtermatch]
                gamma = np.array([calcGamma(self.bandpass, m, self.photparam) 
                                  for m in obs_m5[indexlc][filtermatch]])
                merr, _= calcMagError_m5(mag, bandpass, obs_m5[indexlc], 
                                            photparam, gamma=gamma)

                total_lc = pd.DataFrame([epochs,
                                        mag,
                                        merr,
                                        filters], columns=['time', 'mag','merr','filter']).to_csv(np.join([self.obs_path+'/obs_',
                                                                                                          lcname]))
            return nobj