import numpy as np
import matplotlib.pyplot as plt
import corner
from enterprise_extensions import blocks
from enterprise.signals import gp_signals
from enterprise.signals import signal_base
# import new_blocks
import nonstat_blocks as new_blocks
import pickle, json, os
from sampler import setup_sampler
from hypermodel import HyperModel
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import cgw_model
from fakepta.fake_pta import copy_array
from enterprise_extensions.deterministic import compute_eccentric_residuals
from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler
from scipy.stats import norm

class NestedPTA:

    def __init__(self, psrs, pta, noisedict):

        self.psrs = psrs
        self.pta = pta
        self.param_names = self.pta.param_names
        self.noisedict = noisedict
        self.ndim = len(pta.param_names)
        self.pta.set_default_params(self.noisedict)
        self.prior_ranges = self.get_prior_ranges()
        self.aprior_ranges = np.array([self.prior_ranges[pname] for pname in [*self.prior_ranges]])

    def draw_sample(self):

        x = np.array([p.sample() for p in self.pta.params])
        return x

    def get_prior_ranges(self):

        prior_ranges = {}
        for p in self.pta.params:
            if 'pmin' in [*p.prior._defaults]:
                prior_ranges[p.name] = [p.prior._defaults['pmin'], p.prior._defaults['pmax']]
            elif 'mu' in [*p.prior._defaults]:
                prior_ranges[p.name] = [p.prior._defaults['mu'], p.prior._defaults['sigma']]
        return prior_ranges

    def uniform_prior(self, param_name):
        
        p = 1 / (self.prior_ranges[param_name][1] - self.prior_ranges[param_name][0])
        ln_p = np.log(p)
        return ln_p

    def get_lnprior_weight(self):

        ln_prior = 0.
        for pname in self.param_names:
            ln_prior += self.uniform_prior(pname)
        return ln_prior

    def get_lnprior(self, params):

        return self.pta.get_lnprior(params)

    def get_lnlikelihood(self, params):

        return self.pta.get_lnlikelihood(params)

    def get_lnprob(self, params):

        return self.get_lnlikelihood(params) + self.get_lnprior(params)

    def transform_prior(self, x):

        n = 0
        for pname in self.param_names:
            if ('par_0' in pname) or ('par_1' in pname):
                x[n] = self.gauss_transform(x[n])
                n += 1
            else:
                x[n] *= self.prior_ranges[pname][1] - self.prior_ranges[pname][0]
                x[n] += self.prior_ranges[pname][0]
                n += 1
        return x

    def gauss_transform(self, u):

        t = norm.ppf(u)  # convert to standard normal
        return t

    def get_evidence(self, outfile=None, n=1000, dlogz=0.1):

        sampler = dynesty.NestedSampler(self.get_lnlikelihood, self.transform_prior, len(self.param_names))
        sampler.run_nested(dlogz=dlogz)
        dyplot.cornerplot(sampler.results, color='blue', truths=np.zeros(2),
                           truth_color='black', show_titles=True,
                           max_n_ticks=3, quantiles=None)
        plt.show()
        pickle.dump(sampler.results(), open(outfile, 'wb'))
        json.dump(sampler.results, open('dict_'+outfile, 'wb'))
        return sampler.results['logz'][-1]


# pickle_name = 'GWB-EPTA-DR2_v1.3_newsys_trim'
# noisedict = json.load(open('/home/falxa/noises/noisedict_dr2_newsys_trim.json', 'rb'))
# custom_models = json.load(open('/home/falxa/noises/custom_models_newsys_trim.json', 'rb'))

# pickle_name = 'psrs_epta_dr2_1_3_de440'
# noisedict = json.load(open('/home/falxa/noises/noisefile_epta_dr2_v1_3.json', 'rb'))
# custom_models = json.load(open('/home/falxa/noises/custom_models_final.json'))

pickle_name = 'epta_dr2_full'
noisedict = json.load(open('/home/falxa/noises/dr2full_noisedict.json', 'rb'))
custom_models = json.load(open('/home/falxa/noises/dr2full_custom_models.json'))

pickle_name = 'J1713+0747'
custom_models[pickle_name] = {'RN':30, 'DM':100, 'Sv':None}

psrs = pickle.load(open('/work/falxa/EPTA/pkl/'+pickle_name+'.pkl', 'rb'))
# for psr in psrs_0:
#     custom_models[psr.name] = {'RN':None, 'DM':None, 'Sv':None}


gwb = False
nonstat_gwb = True
orf = 'hd'
crn_psd = 'powerlaw'
hyp = True
psd = 'powerlaw'
rn_psd = 'powerlaw'
dm_psd = 'powerlaw'
order = 1
crn_components = 30
expdip = True
vary_wn = True

N = int(1e6)

# psrdir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/'+psd+'/'+str(order)

for psr in [psrs]:

    # if 'J1713' not in psr.name:
    #     continue

    for nonstat in [False, True]:

        # if nonstat is True:
        #     expdip = False
        # else:
        #     expdip = True
        pta = cgw_model.model_spa(psr, upper_limit=False, vary_wn=vary_wn, J1713_expdip=expdip, custom_models=custom_models, noisedict=noisedict, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=nonstat, order=order, spline=True)
        print(pta.param_names)
        ndim = len(pta.param_names)

        savedir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/'+psd+'_dynesty/'+str(order)+'/'+str(psr.name) + '_expdip_fixed/'
        # savedir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/ecc_ET_2nHz_wn_gwb/nohyp_'+str(psd)+'/'
        outdir = savedir + psr.name + '/' + str(nonstat)
        os.makedirs(outdir, exist_ok=True)

        print(pta.param_names)
        ndim = len(pta.param_names)

        npta = NestedPTA(psr, pta, noisedict)

        nlive = 1024      # number of (initial) live points
        bound = 'multi'   # use MutliNest algorithm
        sample = 'rwalk'  # use the random walk to draw new samples
        ndims = npta.ndim         # two parameters

        # Now run with the static sampler
        sampler = NestedSampler(npta.get_lnlikelihood, npta.transform_prior, ndims,
                                bound=bound, sample=sample, nlive=nlive)
        sampler.run_nested(dlogz=0.1)

        res = sampler.results

        logZdynesty = res.logz[-1]        # value of logZ
        logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

        # output marginal likelihood
        # print('Marginalised evidence (using static sampler) is {} ± {}'.format(logZdynesty, logZerrdynesty))

        # get the posterior samples
        weights = np.exp(res['logwt'] - res['logz'][-1])
        postsamples = resample_equal(res.samples, weights)
        logl = res.logl

        np.savetxt(outdir+'/logZdynesty.txt', np.array([logZdynesty, logZerrdynesty]))
        np.savetxt(outdir+'/samples.txt', np.array(postsamples))
        np.savetxt(outdir+'/weights.txt', np.array(weights))
        np.savetxt(outdir+'/params.txt', pta.param_names, fmt='%s')
        np.savetxt(outdir+'/logl.txt', logl, fmt='%s')

        try:
            corner.corner(postsamples, labels=pta.param_names, color='C0', hist_kwargs={'density': True})
            plt.savefig(outdir+'/corner_plot.png')
            plt.clf()
            plt.cla()
        except:
            print('chibropracteur')
