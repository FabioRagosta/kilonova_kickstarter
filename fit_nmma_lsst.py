#general modules
import numpy as np
import pandas as pd
import os, sys, time, glob
from astropy.time import Time
from scipy.special import logsumexp
import copy
import bilby

#nmma modules
from nmma.em.model import SimpleKilonovaLightCurveModel, GRBLightCurveModel, GenericCombineLightCurveModel
from nmma.em import training, utils, model_parameters, analysis
from nmma.em.likelihood import OpticalLightCurve
from nmma.em.utils import getFilteredMag




class fit_lc(object):
    """
        Class object, observed lightcurve from LSST observing constraints
        Parameters
        ------------
            model_names= list of light curves models from nmma (DEFAULT: ['TrPi2018','Me2017'] )
            sample_times= array with the light curve phases      (example = tmin, tmax, dt = 0.5, 20.0, 0.5
                                                                sample_times = np.arange(tmin, tmax + dt, dt))
            threshold= the threshold between the opacity of the the components if n_kn_component >1 (DEFAULT = None)
            priors_path = path of the priors file (DEFAULT = '~/nmma/priors')
            data_path = path where the save the csv of the observed light curves   (DEFAULT = ./obs_lc)
            output = path where the save the posteriors file   (DEFAULT = ./posterior)
            detection_limit = number of light curves to simulate (DEFAULT = {'u':24,'g':25,'r':25,'i':24,'z':23,'y':22})
             
        Returns
        -------
            files with the posterior distribution of the parameters from the fit
        """
    def __init__(self, model_names = ['TrPi2018','Me2017'],data_path = './obs_lc',
                 priors_path = '/Users/fabioragosta/nmma/priors',sample_times=sample_times, 
                 Ebv_max=0.5724, error_budget=0.1, seed=42,output='./posteriors',
                 detection_limit={'u':24,'g':25,'r':25,'i':24,'z':23,'y':22},**kwargs):
        self.model_names = model_names
        self.priors_path = priors_path
        self.data_path =data_path
        self.sample_times = sample_times
        self.Ebv_max = Ebv_max
        self.error_budget=error_budget
        self.detection_limit=detection_limit
        self.output=output
        self.seed = seed
        models = []
        for model_name in self.model_names:
            if model_name == "TrPi2018":
                lc_model = GRBLightCurveModel(
                    sample_times=sample_times,
                    resolution=12,
                    jetType=0,
                )

            elif model_name == "nugent-hyper":
                lc_model = SupernovaLightCurveModel(
                    sample_times=sample_times, model="nugent-hyper"
                )

            elif model_name == "salt2":
                lc_model = SupernovaLightCurveModel(
                    sample_times=sample_times, model="salt2"
                )

            elif model_name == "Piro2021":
                lc_model = ShockCoolingLightCurveModel(sample_times=sample_times)

            elif model_name == "Me2017" or model_name == "PL_BB_fixedT":
                lc_model = SimpleKilonovaLightCurveModel(sample_times=sample_times,
                                                         model=model_name)

            else:
                print('set the model!')
            
        models.append(lc_model)    
        
        if len(models) > 1:
            self.light_curve_model = GenericCombineLightCurveModel(models_name, sample_times)
        else:
            self.light_curve_model = models[0]
            
        if not os.path.exists(self.output):
            os.mkdir('./posteriors')
            
    def load_Event(self,filename):
        lines = [line.rstrip("\n") for line in open(filename)]
        lines = filter(None, lines)
        data = {}
        for i,line in enumerate(lines):
            if i==0:
                continue
            lineSplit = line.split(",")
            lineSplit = list(filter(None, lineSplit))
            mjd = Time(float(lineSplit[1]), format='jd').mjd
            filt = lineSplit[2]
            mag = float(lineSplit[3])
            dmag = float(lineSplit[4])
            flag=float(lineSplit[-1])
            if filt not in data:
                data[filt] = np.empty((0, 3), float)
            if flag==1:
                data[filt] = np.append(data[filt], np.array([[mjd, mag, dmag]]), axis=0)
        return data
                    
    def fit(self):
        datalist = glob.glob(self.data_path+'/*')
        for datafile in datalist:
            filename= datafile.split('/')[-1]
            filename = filename[0:-4]
            trigger_time = np.amin(pd.read_csv(datafile)['time'])
            filters = np.unique(pd.read_csv(datafile)['filter'])
            data = self.load_Event(datafile)
            likelihood_kwargs = dict(
                light_curve_model=self.light_curve_model,
                filters=filters,
                light_curve_data=data,
                trigger_time=trigger_time,
                tmin=np.amin(self.sample_times),
                tmax=np.amax(self.sample_times),
                error_budget=self.error_budget,
                detection_limit=self.detection_limit,
                )
            likelihood = OpticalLightCurve(**likelihood_kwargs)

            priors = bilby.gw.prior.PriorDict(self.priors_path+'/Me2017.prior')
            result = bilby.run_sampler(
                likelihood=likelihood,
                priors=priors,
                outdir=self.output,
                label=filename,
                seed=self.seed
                )
            result.save_posterior_samples(filename='posteriors_'+filename+'.dat')
        return print('fit completes!')
