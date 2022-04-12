import numpy as np
import pandas as pd
import os, sys, time, glob
from scipy.special import logsumexp
import copy
#nmma modules
sys.path.append('/Users/fabioragosta/nmma')
import nmma
from nmma.em.model import SimpleKilonovaLightCurveModel,GRBLightCurveModel
import nmma.em.utils as utils



class GRBKN_pop(object):
    def __init__(self,priors, sample_times, threshold=None,n_kn_component=1, data_path='./lc', n_lc=1000, filters = ["u", "g", "r", "i", "z", "y", "J", "H", "K"],**kwargs):
        self.priors = priors #dictionary with parameters distributions 
        self.sample_times = sample_times
        self.n_kn_component = n_kn_component
        self.n_lc = n_lc
        self.filters = filters
        self.data_path = data_path
        self.threshold=threshold
        if not os.path.exists(self.data_path):
            os.mkdir('./lc')
    def KN_lc(self, sample_times, params, treshold=None):
        #definire treshold
        if self.n_kn_component!=1:
            if not treshold:
                print('################')
                print('ERROR: set the threshold for k')
                print('################')
            try:
                params.update({"log10_kappa_r":np.random.choice(self.prior['log10_kappa_r'][:,0][self.prior['log10_kappa_r']<treshold], p=None)})
                kn_model = SimpleKilonovaLightCurveModel(sample_times=sample_times)
                _, mag1 = kn_model.generate_lightcurve(sample_times, params)
                params.update({"log10_kappa_r":np.random.choice(self.prior['log10_kappa_r'][:,0][self.prior['log10_kappa_r']>treshold], p=None)})
                kn_model = SimpleKilonovaLightCurveModel(sample_times=sample_times)
                _, mag2 = kn_model.generate_lightcurve(sample_times, params)
                mag = pd.DataFrame(columns=mag1.keys)
            except ImportError:
                print("problems with multi-component ejecta for KN model")
                
            for f in mag.keys:
                mag[f] = total_mag_kn = (
                    -5.0
                    / 2.0
                    * logsumexp(
                        [ -2.0 / 5.0 * ln10 * mag[f],-2.0 / 5.0 * ln10 * mag_[f]],
                        axis=0,
                    )
                    / ln10
                )
        else:
            kn_model = SimpleKilonovaLightCurveModel(sample_times=sample_times)
            _, mag = kn_model.generate_lightcurve(sample_times, params)
        return mag
    
    def GRB_lc(self, sample_times, params):
        grb_model = GRBLightCurveModel(sample_times=sample_times)
        _, mag = grb_model.generate_lightcurve(sample_times, params)
        return mag
    
    def run(self):
        ln10 = np.log(10)
        #Loop starts
        n = 0
        while n!=self.n_lc:
            n+=1
            #Check probability distributions
            #Opacity
            if self.priors['log10_kappa_r'].ndim==2:          
                k = np.random.choice(self.priors['log10_kappa_r'][:,0], p=self.priors['log10_kappa_r'][:,1])
            else:
                k = np.random.choice(self.priors['log10_kappa_r'], p=None)
            #Ejecta mass
            if self.priors['log10_Mej'].ndim==2:
                Mej = np.random.choice(self.priors['log10_Mej'][:,0], p=self.priors['log10_Mej'][:,1])
            else:
                Mej = np.random.choice(self.priors['log10_Mej'], p=None)
            #Ejecta velocity
            if self.priors['log10_vej'].ndim==2:
                vej = np.random.choice(self.priors['log10_vej'][:,0], p=self.priors['log10_vej'][:,1])
            else:
                vej = np.random.choice(self.priors['log10_vej'], p=None)
            #Luminosity distance
            if self.priors['luminosity_distance'].ndim==2:
                ld = np.random.choice(self.priors['luminosity_distance'][:,0], p=self.priors['luminosity_distance'][:,1])
            else:
                ld = np.random.choice(self.priors['luminosity_distance'], p=None)        
            #parameters are set for the simulation
            params = {
                "luminosity_distance": ld,
                "beta": self.priors['beta'],
                "log10_kappa_r": k,
                "KNtimeshift": self.priors['KNtimeshift'],
                "log10_vej": vej,
                "log10_Mej": Mej,
                "Ebv": self.priors['Ebv'],
                "log_likelihood": self.priors['log_likelihood'],
                "log_prior": self.priors['log_prior'],
                'jetType':self.priors['jetType'],
                "inclination_EM":self.priors['inclination_EM'],
                "log10_E0": self.priors['log10_E0'],
                "thetaCore": self.priors['thetaCore'],
                "thetaWing": self.priors['thetaWing'],
                "log10_n0": self.priors['log10_n0'],
                'p':           self.priors['p'],    # electron energy distribution index
                "log10_epsilon_e":   self.priors['log10_epsilon_e'],    # epsilon_e
                "log10_epsilon_B":   self.priors['log10_epsilon_B'],   # epsilon_B
                'xi_N':        self.priors['xi_N'],
                }
            #lc generation
            mag_kn = self.KN_lc(sample_times, params, self.threshold)
            mag_grb = self.GRB_lc(sample_times, params)
            total_mag = pd.DataFrame(columns=self.filters)
            for f in self.filters:
                total_mag[f] = (
                    -5.0
                    / 2.0
                    * logsumexp(
                        [-2.0 / 5.0 * ln10 * mag_grb[f], -2.0 / 5.0 * ln10 * mag_kn[f]],
                        axis=0,
                    )
                    / ln10) +5*np.log10(ld*1e6/10)
            total_mag['time'] = self.sample_times
            total_mag.to_csv(self.data_path+'/Mej{0:.2f}_vej{1:.2f}_k{2:.2f}.csv'.format(Mej,vej,k))
        return print('End of the simulation!')
