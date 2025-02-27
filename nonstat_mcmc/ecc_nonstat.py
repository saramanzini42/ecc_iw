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
from enterprise_extensions.deterministic import compute_eccentric_residuals
from fakepta.fake_pta import Pulsar, copy_array, make_fake_array
from fakepta import correlated_noises
import scipy.constants as sc
import time

# pickle_name = 'GWB-EPTA-DR2_v1.3_newsys_trim'
# noisedict = json.load(open('/home/falxa/noises/noisedict_dr2_newsys_trim.json', 'rb'))
# custom_models = json.load(open('/home/falxa/noises/custom_models_newsys_trim.json', 'rb'))

# pickle_name = 'psrs_epta_dr2_1_3_de440'
# noisedict = json.load(open('/home/falxa/noises/noisefile_epta_dr2_v1_3.json', 'rb'))
# custom_models = json.load(open('/home/falxa/noises/custom_models_final.json'))

# psrs_0 = pickle.load(open('/work/falxa/EPTA/pkl/'+pickle_name+'.pkl', 'rb'))
# for psr in psrs_0:
#     custom_models[psr.name] = {'RN':None, 'DM':None, 'Sv':None}

terr = -7.
psrs = make_fake_array(npsrs=10, Tobs=15, ntoas=500, gaps=False, toaerr=10**terr, pdist=None, freqs=[1400], isotropic=True, backends=['mizel'], gp_noises=False)
# psrs = copy_array(psrs_0, noisedict, custom_models)
seed = int(time.time())
np.random.seed(seed)
# np.random.seed(int(1703900445))

noisedict = {}
custom_models = {}
for psr in psrs:
    psr.make_ideal()
    psr.custom_model = {'RN':None, 'DM':None, 'Sv':None}
    noisedict.update(psr.noisedict)
    custom_models[psr.name] = {'RN':None, 'DM':None, 'Sv':None}


# psr = Pulsar(np.linspace(0., 15*365.25*3600*24, 500), 10**(-7), 1.5, 3., 1., custom_noisedict=None, custom_model={'RN':10, 'DM':None, 'Sv':None}, backends=['bakend'])
# psr.init_noisedict()
# psr.make_ideal()
# # psr.add_white_noise()
# psr.add_red_noise(gp=True, log10_A=-14.5, gamma=4.33)

# rn = psr.residuals.copy()

# cgw params
T = 15 * 365.25 * 24 * 3600
log10_h = -13.5
log10_dist = 2.
cos_inc = 0.3
l0 = 1.
# l0 = 2.
# gamma0 = 2.6
gamma0 = 3.
psi = 0.6
log10_mc = 9.2
e0 = 0.5
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
orf = None
hyp = False
psd = 'powerlaw'
rn_psd = 'powerlaw'
dm_psd = 'powerlaw'
order = 2

N = int(1e5)

# psrdir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/'+psd+'/'+str(order)

if gwb:
    psrs = [psrs]

for psr in psrs:

    for log10_h in [params['log10_h']]:

        for psr_i in psr:
            cgw = compute_eccentric_residuals(psr_i.toas, psr_i.theta, psr_i.phi, cos_gwtheta, gwphi,
                                        log10_mc, log10_dist, log10_h, log10_F, cos_inc,
                                        psi, gamma0, e0, l0, q, nmax=400, pdist=1.0,
                                        pphase=None, pgam=None, psrTerm=False)
            plt.plot(psr_i.toas, cgw)
            psr_i.residuals += cgw
            psr.add_white_noise()

        if gwb:
            if log10_h == -50.:
                par_0 = np.zeros(order)
                par_1 = np.zeros(order)
                for psr_i in psr:
                    psr_i.add_white_noise()
            else:
                par_0 = None
                par_1 = None
            pta_nonstat = cgw_model.model_gw(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict, orf=orf, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=nonstat_gwb, order=order, spline=True, par_0=par_0, par_1=par_1)
        else:
            pta_nonstat = cgw_model.model_spa(psr, upper_limit=False, J1713_expdip=True, custom_models=custom_models, noisedict=noisedict, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=nonstat_gwb, order=order, spline=True)
        if hyp:
            if gwb:
                pta_stat = cgw_model.model_gw(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict, orf=orf, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=False, order=order)
            else:
                pta_stat = cgw_model.model_spa(psr, upper_limit=False, J1713_expdip=True, custom_models=custom_models, noisedict=noisedict, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=False, order=order)
            ptas = {}
            ptas[0] = pta_nonstat
            ptas[1] = pta_stat
            pta = HyperModel(ptas)
        else:
            pta = pta_nonstat
        print(pta.param_names)
        ndim = len(pta.param_names)

        x = np.hstack([p.sample() for p in pta_nonstat.params])
        
        if hyp:
            x = np.append(x, np.random.uniform(-0.5, 1.5))

        if gwb:
            pickle_name = 'sim_psrs'
            savedir = '/work/falxa/EPTA/nonstat/ecc_sim/'+pickle_name+'/'+str(orf)+'/'+str(seed)+'/gwb_'+str(log10_A)+'_'+str(gamma)+'_ecc_'+str(e0)+'_ET_'+str(round(log10_F, 2))+'nHz_log10_h'+str(log10_h)+'wn_'+str(terr)+'_gwb'
            # outdir = savedir + str(orf)
            outdir += '/order_'+str(order)
            outdir = savedir
        os.makedirs(outdir, exist_ok=True)
        ndim = len(pta.param_names)

        plt.savefig(savedir + '/ecc.png')
        plt.cla()
        plt.clf()

        json.dump(params, open(savedir+'/ecc_params.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        if hyp:
            sampler = pta.setup_sampler(outdir=outdir, resume=False, sample_nmodel=True, groups=None
                            #, empirical_distr='/home/falxa/noises/emp/cw_search_-8.6_-8.0_patch_None_custom_models_ncgw_0_newsys_trim.pkl'
                            )
        else:
            sampler =  setup_sampler(pta, outdir=outdir, resume=False,
                            empirical_distr=None, groups=None, human=None,
                            save_ext_dists=False, loglkwargs={}, logpkwargs={})

        # sampler for N steps
        sampler.sample(x, N, SCAMweight=50, AMweight=0, DEweight=50)

        chains = np.genfromtxt(outdir+'/chain_1.txt')
        burn = int(0.1*len(chains))
        chains = chains[burn:]
        corner.corner(chains[:, :-4], labels=pta.param_names)
        plt.savefig(savedir + '/gwb_nonstat.png')
        plt.cla()
        plt.clf()