import numpy as np
import pandas as pd
import os, sys, time, glob
import lsst.sims.maf.metrics as metrics
from lsst.sims.utils import uniformSphere
import lsst.sims.maf.slicers as slicers
from lsst.sims.photUtils import Dust_values
from lsst.sims.photUtils import Bandpass, SignalToNoise, PhotometricParameters, calcMagError_m5, calcGamma

class GRBKN_obs(metrics.BaseMetric):
    """
        Class object, observed lightcurve from LSST observing constraints
        Parameters
        ------------
            mjdCol= MJD observations column name from Opsim database      (DEFAULT = observationStartMJD) 
            m5Col= Magnitude limit column name from Opsim database      (DEFAULT = fiveSigmaDepth)
            filterCol= Filters column name from Opsim database      (DEFAULT = filter)
            exptimeCol = Column name for the total exposure time of the visit(DEFAULT = visitExposureTime)
            nightCol = The night's column of the survey (starting at 1) (DEFAULT = night)
            vistimeCol = Column name for the total time of the visit (DEFAULT = visitTime)
            RACol= RA column name from Opsim database      (DEFAULT = fieldRA)
            DecCol= Dec column name from Opsim database      (DEFAULT = fieldDec)
            surveyduration= Survey Duration      (DEFAULT = 10)
            mjd0= Survey start date      (DEFAULT = 59853.5)
            Filter_selection = 
        Returns
        -------
            nobj: number of detected lightcurves
            csv files with the observed lightcurves
        """
    def read_lightCurve(self, file):
        """Reads in a csv file, from the simulated ligh curves, time and mag columns for each filter
        Returns
        -------
        numpy.ndarray
            The data read from the ascii text file, in a numpy structured array with columns
            'ph' (phase / epoch, in days), 'mag' (magnitude), 'flt' (filter for the magnitude).
        """
        time = np.arange(0.5, 20 + 0.5, 0.5)
        ll = pd.DataFrame(file, columns=['file_paths'])
        ll['lc']=ll['file_paths'].apply(lambda x: pd.read_csv(x, index_col=False))
        ll['transDuration'] = ll['lc'].apply(lambda x: time.max() - time.min())
        ll['T0'] = ll['lc'].apply(lambda x: time.min())
        return ll
    
    def __init__(self, metricName='GRBKN_obs', mjdCol='observationStartMJD', 
                 RACol='fieldRA', DecCol='fieldDec',filterCol='filter', m5Col='fiveSigmaDepth', 
                 exptimeCol='visitExposureTime',nightCol='night',vistimeCol='visitTime', snrlim=5, ptsNeeded=2, dist_Mpc=46.48, mjd0=59853.5,
                 data_path='./lc', obs_path='./obs_lc',surveyduration=10,Filter_selection = False,nFilter=1, save=True,**kwargs):
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
        self.dist_Mpc = dist_Mpc
        self.surveyduration=surveyduration
        self.mjd0=mjd0
        self.Filter_selection=Filter_selection
        self.nFilter=nFilter
        self.snrlim = snrlim
        self.save = save
        if not os.path.exists(self.obs_path):
            os.mkdir(self.obs_path)
        self.bandpass = Bandpass(wavelen=np.array([480.2,623.1,754.2]),
                                 sb=np.array([0.1,0.1,0.1]),wavelen_min=380, wavelen_max=850, wavelen_step=10)
        super(GRBKN_obs, self).__init__(col=[self.mjdCol,self.m5Col, self.filterCol,self.RACol,
                                                                   self.DecCol,self.exptimeCol,self.nightCol,
                                                                   self.vistimeCol],
                                                       metricDtype='object', units='',
                                                       metricName=metricName, **kwargs)
        self.bandpass = Bandpass(wavelen=np.array([480.2,623.1,754.2]),sb=np.array([0.1,0.1,0.1]),wavelen_min=380, wavelen_max=850, wavelen_step=10)
        self.photparam = PhotometricParameters() 
        lcfile = glob.glob(self.data_path+'/*.csv')
        lcfile = np.array(self.lcfile).reshape(-1)
        self.lcs = self.read_lightCurve(flc)
     
    def make_lightCurve(self, template, time, filters,transduration):
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
        time = (time-self.mjd0)%transduration
        for key in set(flt):
            # Interpolate the lightcurve template to the times of the observations, in this filter.
            temp_ph=np.array(np.logspace(-1,np.log10(30),50),float)#template['time'], float)
            lcMags[filters==key] = np.interp(time[filters==key], temp_ph,
                                        np.array(template[key], float))
        return lcMags
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

    def name_lc(self, t, lcname, name):
        return self.obs_path+'/obs_{}_'.format(name)+lcname+'_T={}.npy'.format(t)
    def run(self,dataSlice, slicePoint=None):
        # Sort the entire dataSlice in order of time. 
        dataSlice.sort(order=self.mjdCol)
        dataSlice = self.coadd(pd.DataFrame(dataSlice))
        
        field_idx = np.random.choice(range(np.size(dataSlice[self.RACol])))
        Ra, Dec = dataSlice[self.RACol][field_idx],dataSlice[self.DecCol][field_idx]
        bandpass = self.bandpass
        photparam = self.photparam
        calcSNR_m5=np.vectorize(SignalToNoise.calcSNR_m5)
        
        obs_filter = dataSlice[self.filterCol]
        obs = dataSlice[self.mjdCol]       
        obs_m5 = dataSlice[self.m5Col]
        nobj = np.array([0,0],[('detected','i4'),('undetected','i4')]) 
        

        lcNumberStart = -1 * np.floor((dataSlice[self.mjdCol].min() - self.mjd0) / self.lcs['transDuration'].to_numpy()[:,None])
        # Calculate the time/epoch for each lightcurve.
        lcEpoch = (obs-self.mjd0) % self.lcs['transDuration'].to_numpy()[:,None]
        # Identify the observations which belong to each distinct light curve.
        lcNumber = np.floor((obs-self.mjd0 )  / lcs['transDuration'].to_numpy()[:,None]) + lcNumberStart
        lcNumberStart = lcNumber.max(axis=0)
        ulcNumber = np.array(list(map(np.unique,lcNumber)))
        lcLeft = np.array([np.searchsorted(lcNumber[i,:], ulcNumber[i,:], side='left')for i in range(ulcNumber.shape[0])])
        lcRight = np.array([np.searchsorted(lcNumber[i,:], ulcNumber[i,:], side='right')for i in range(ulcNumber.shape[0])])
        self.lcs['obs'] = self.lcs.apply(lambda x: self.make_lightCurve(x['lc'],obs, obs_filter,x['transDuration']), axis=1)
        lcMag = np.array(self.lcs['obs'].tolist())  + 5*np.log10(self.dist_Mpc*10**6)-5
        gamma = np.array([calcGamma(self.bandpass, m, self.photparam) 
                          for m in obs_m5])

        merr, _= calcMagError_m5(lcMag, bandpass, obs_m5,
                                    photparam, gamma=gamma)
        #lcSNR,_ = calcSNR_m5(lcMag, bandpass, obs_m5, photparam)
        lcpoints_AboveThresh = np.zeros(np.shape(lcMag), dtype=bool) 
        for f in np.unique(obs_filter):                    
            filtermatch = np.where(obs_filter == f)
            lcpoints_AboveThresh[:,filtermatch] = np.where(lcMag[:,filtermatch] <= obs_m5[filtermatch],True,lcpoints_AboveThresh[:,filtermatch])
        T0 = lcs['T0'][:,None] + self.mjd0 + ulcNumber*lcs['transDuration'].to_numpy()[:,None]

        #INITIALIZATION 10yrs LCs
        detection_map = np.array([[lcpoints_AboveThresh[i,init:end] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc))],dtype=object)
        lcEpochs= np.array([[lcEpoch[i,init:end] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc)) ],dtype=object)
        lcMags = np.array([[lcMag[i,init:end] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc)) ],dtype=object)
        lcMerr = np.array([[merr[i,init:end] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc)) ],dtype=object)
        lcfilters = np.array([[obs_filter[init:end] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc)) ],dtype=object)
        lcmaglim = np.array([[obs_m5[init:end] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc)) ],dtype=object)

        Dpoints = np.array([[np.sum(detections, axis= 0) for detections in detection_map[i]] for i in range(len(flc))])  #counts the number of detected points
        nobj['detected']=np.sum([points>self.ptsNeeded for i in range(len(flc))  for points in Dpoints[i] ])
        nobj['undetected']=np.sum([points>self.ptsNeeded for i in range(len(flc))  for points in Dpoints[i] ])
        if self.save:
            shift = np.array([[ulcNumber[i,j]*lcs['transDuration'][i] for j,(init,end) in enumerate(zip(lcLeft[i,:],lcRight[i,:])) if init!= end]for i in range(len(flc)) ])
            epochs = lcEpochs +  self.mjd0 + shift

            total_lc = np.dstack([epochs,
                    lcfilters,
                    lcMags,
                    lcMerr,
                    lcmaglim,
                    detection_map])

            name_ = np.vectorize(self.name_lc)
            names = name_(T0,
                          np.repeat(np.array(flc)[:,None],ulcNumber.shape[1],axis=0).reshape(ulcNumber.shape),
                          ulcNumber)
            names = np.expand_dims(names, axis=-1)
            out = list(zip(names,total_lc))
            with open(self.obs_path+'/simRA={}_DEC={}_part{}.pkl'.format(np.round(Ra,2),np.round(Dec,2), nflc+1), 'wb') as outfile:
                pickle.dump(out, outfile)
            ndet = np.sum(Dpoints)
            res = ndet/nobj['detected'][0]/6
            print(res)
        else:
            ndet = np.sum(Dpoints)
            res = ndet/nobj['detected'][0]/6
            print(res)
        return res
