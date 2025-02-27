import os, pickle, json
import sys
from scipy.special import ndtri
import numpy as np
from scipy.stats import norm
from fakepta.fake_pta import Pulsar, copy_array, make_fake_array
import scipy.constants as sc
import scipy.interpolate as interpolate
from scipy.fft import rfft, irfft
from fakepta import correlated_noises
from enterprise_extensions.deterministic import compute_eccentric_residuals
import time
import optparse

# import dynesty (we'll give an example with both the static and
# dynamic nested samplers)
from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler
import cgw_model
from enterprise_extensions import blocks
from enterprise.signals import signal_base
import matplotlib.pyplot as plt
import corner
from enterprise.signals import deterministic_signals
from enterprise.signals import parameter

parser = optparse.OptionParser()
parser.add_option('--ecc', action='store', dest='e', default=0.)
(options, args) = parser.parse_args()

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
            if ('gw_a_amp' in pname) or ('gw_b_gam' in pname):
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

terr = -7.
psrs = make_fake_array(npsrs=10, Tobs=15, ntoas=500, gaps=False, toaerr=10**terr, pdist=None, freqs=[1400], isotropic=True, backends=['mizel'], gp_noises=False)
# psrs = copy_array(psrs_0, noisedict, custom_models)
seed = int(time.time())
seed = int(1704049979)
np.random.seed(seed)
# np.random.seed(int(1703900445))

noisedict = {}
custom_models = {}
for psr in psrs:
    psr.make_ideal()
    psr.custom_model = {'RN':None, 'DM':None, 'Sv':None}
    noisedict.update(psr.noisedict)
    custom_models[psr.name] = {'RN':None, 'DM':None, 'Sv':None}

# cgw params
T = 15 * 365.25 * 24 * 3600
log10_h = -14.
log10_dist = 2.
cos_inc = 0.3
l0 = 1.
# l0 = 2.
# gamma0 = 2.6
gamma0 = 3.
psi = 0.6
log10_mc = 9.2
e0 = float(options.e)
q = 1.
log10_F = np.log10(1/T)
# cgw virgo
cos_gwtheta = 0.12
gwphi = 3.2
# # cgw fornax
# params['costheta'] = -0.57
# params['phi'] = 0.95

params = {}
params['log10_h'] = log10_h
params['log10_dist'] = log10_dist
params['cos_inc'] = cos_inc
params['l0'] = l0
params['gamma0'] = gamma0
params['psi'] = psi
params['log10_mc'] = log10_mc
params['e0'] = e0
params['q'] = q
params['log10_F'] = log10_F
params['cos_gwtheta'] = cos_gwtheta
params['gwphi'] = gwphi

# psrs = [psr]
custom_models = {}
noisedict = {}
for psr in psrs:
    custom_models[psr.name] = psr.custom_model
    noisedict.update(psr.noisedict)

log10_A = -14.2
gamma = 3.
correlated_noises.add_correlated_red_noise_gp(psrs, orf='hd', log10_A=log10_A, gamma=gamma, rn_components=100)

gwb = True
nonstat_gwb = True
orf = 'hd'
# orf = None
hyp = False
psd = 'powerlaw'
rn_psd = 'powerlaw'
dm_psd = 'powerlaw'
order = 1

# psrdir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/'+psd+'/'+str(order)

if gwb:
    psrs = [psrs]

for psr in psrs:

    for log10_h in [params['log10_h']]:

        if log10_h == -50:
            ns = False
        else:
            ns = True

        for psr_i in psr:
            plt.plot(psr_i.toas, psr_i.residuals, linewidth=2)

        for psr_i in psr:
            cgw = compute_eccentric_residuals(psr_i.toas, psr_i.theta, psr_i.phi, cos_gwtheta, gwphi,
                                        log10_mc, log10_dist, log10_h, log10_F, cos_inc,
                                        psi, gamma0, e0, l0, q, nmax=400, pdist=1.0,
                                        pphase=None, pgam=None, psrTerm=False)
            plt.plot(psr_i.toas, cgw,  linestyle='--')
            psr_i.residuals += cgw
            psr_i.add_white_noise()

        if not ns:
            par_0 = np.zeros(order)
            par_1 = np.zeros(order)
            for psr_i in psr:
                psr_i.add_white_noise()
        else:
            par_0 = None
            par_1 = None

        # set psr number and model
        savedir = '/work/falxa/EPTA/nonstat/ecc_sim/sim_psrs'
        outdir_0 = savedir + '/dynesty/h14_'+str(seed)+'/'
        outdir_0 += 'ecc_'+str(params['e0'])+'/'
        os.makedirs(outdir_0, exist_ok=True)

        nonstats = [False, True]
        psds = ['powerlaw', 'powerlaw']
        fyr = 1/sc.Julian_year

        for nstat, psd_i in zip(nonstats, psds):

            str_dir = str(nstat)+'_powerlaw_ecc_logh_' + str(log10_h)
            outdir = outdir_0 + str_dir

            os.makedirs(outdir, exist_ok=True)

            plt.savefig(outdir + '/ecc.png')
            plt.cla()
            plt.clf()

            psd = 'powerlaw_t'
            rn_psd = 'powerlaw'
            dm_psd = 'powerlaw'
            nonstat = True

            ''' SAMPLER '''
            pta = cgw_model.model_gw(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, par_0=par_0, par_1=par_1,
                                            noisedict=noisedict, orf=orf, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=nstat, order=order)
            
            print(pta.param_names)
            ndim = len(pta.param_names)

            npta = NestedPTA(psr, pta, noisedict)

            nlive = 1024      # number of (initial) live points
            bound = 'multi'   # use MutliNest algorithm
            sample = 'rwalk'  # use the random walk to draw new samples
            ndims = npta.ndim         # two parameters

            # Now run with the static sampler
            np.savetxt(outdir+'/params.txt', pta.param_names, fmt='%s')
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
