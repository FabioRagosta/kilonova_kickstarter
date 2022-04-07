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
    def __init__(self, data_path='./lc',priors, sample_times, n_kn_component, n_lc=1000, filters = ["u", "g", "r", "i", "z", "y", "J", "H", "K"]):
        self.priors = priors #dictionary with parameters distributions 
        self.sample_times = sample_times
        self.n_kn_component = n_kn_component
        self.n_lc = n_lc
        self.filters = filters
        self.data_path = data_path
        if not os.path.exist(self.data_path):
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
    
    def run():
        ln10 = np.log(10)
        #Loop starts
        n = 0
        while n!=n_lc:
            n+=1
            #Check probability distributions
            #Opacity
            if np.shape(self.prior['log10_kappa_r'])[1]==2:          
                k = np.random.choice(self.prior['log10_kappa_r'][:,0], p=self.prior['log10_kappa_r'][:,1])
            else:
                k = np.random.choice(self.prior['log10_kappa_r'][:,0], p=None)
            #Ejecta mass
            if np.shape(self.prior['log10_Mej'])[1]==2:
                Mej = np.random.choice(self.prior['log10_Mej'][:,0], p=self.prior['log10_Mej'][:,1])
            else:
                Mej = np.random.choice(self.prior['log10_Mej'][:,0], p=None)
            #Ejecta velocity
            if np.shape(self.prior['log10_vej'])[1]==2:
                vej = np.random.choice(self.prior['log10_vej'][:,0], p=self.prior['log10_vej'][:,1])
            else:
                vej = np.random.choice(self.prior['log10_vej'][:,0], p=None)
            #Luminosity distance
            if np.shape(self.prior['DL'])[1]==2:
                ld = np.random.choice(self.prior['DL'][:,0], p=self.prior['DL'][:,1])
            else:
                ld = np.random.choice(self.prior['DL'][:,0], p=None)        
            #parameters are set for the simulation
            params = {
                "luminosity_distance": ld,
                "beta": self.prior['beta'],
                "log10_kappa_r": k,
                "KNtimeshift": self.prior['KNtimeshift'],
                "log10_vej": vej,
                "log10_Mej": Mej,
                "Ebv": self.prior['Ebv'],
                "log_likelihood": self.prior['log_likelihood'],
                "log_prior": self.prior['log_prior'],
                'jetType':self.prior['jetType'],
                "inclination_EM":self.prior['inclination_EM'],
                "log10_E0": self.prior['log10_E0'].,
                "thetaCore": self.prior['thetaCore'],
                "thetaWing": self.prior['thetaWing'],
                "log10_n0": self.prior['log10_n0'],
                'p':           self.prior['p'],    # electron energy distribution index
                "log10_epsilon_e":   self.prior['log10_epsilon_e'],    # epsilon_e
                "log10_epsilon_B":   self.prior['log10_epsilon_B'],   # epsilon_B
                'xi_N':        self.prior['xi_N'],
                }
            #lc generation
            mag_kn = self.KN_lc(sample_times, params, treshold=None)
            mag_grb = self.GRB_lc(sample_times, params)
            total_mag = pd.DataFrame(columns=filters)
            for f in self.filters:
                total_mag[f] = (
                    -5.0
                    / 2.0
                    * logsumexp(
                        [-2.0 / 5.0 * ln10 * mag_grb[f], -2.0 / 5.0 * ln10 * mag_kn[f]],
                        axis=0,
                    )
                    / ln10
            total_mag['time'] = self.sample_times
            total_mag.to_csv(np.join([self.data_path,'/Mej{}_vej{}_k{}.csv'.format(Mej,vej,k)]))
        return print('End of the simulation!')