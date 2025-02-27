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

pickle_name = 'GWB-EPTA-DR2_v1.3_newsys_trim'
noisedict = json.load(open('/home/falxa/noises/noisedict_dr2_newsys_trim.json', 'rb'))
custom_models = json.load(open('/home/falxa/noises/custom_models_newsys_trim.json', 'rb'))

pickle_name = 'psrs_epta_dr2_1_3_de440'
noisedict = json.load(open('/home/falxa/noises/noisefile_epta_dr2_v1_3.json', 'rb'))
custom_models = json.load(open('/home/falxa/noises/custom_models_final.json'))

pickle_name = 'epta_dr2_full'
noisedict = json.load(open('/home/falxa/noises/dr2full_noisedict.json', 'rb'))
custom_models = json.load(open('/home/falxa/noises/dr2full_custom_models.json'))

psrs_0 = pickle.load(open('/work/falxa/EPTA/pkl/'+pickle_name+'.pkl', 'rb'))
# for psr in psrs_0:
#     custom_models[psr.name] = {'RN':None, 'DM':None, 'Sv':None}

psrs = copy_array(psrs_0, noisedict, custom_models)

tmin = np.amin([psr.toas.min() for psr in psrs])
tmax = np.amax([psr.toas.max() for psr in psrs])
Tspan = tmax - tmin

# keep half of data
alpha = 0.25
tmin_slice = tmin + alpha*Tspan
for psr in psrs:
    mask = psr.toas < (tmin_slice + (0.5 + alpha)*Tspan)
    mask *= psr.toas > tmin_slice
    psr.toas = psr.toas[mask]
    psr.toaerrs = psr.toaerrs[mask]
    psr.residuals = psr.residuals[mask]
    psr.freqs = psr.freqs[mask]
    psr.backend_flags = psr.backend_flags[mask]
    psr.Mmat = psr.Mmat[mask]

gwb = True
nonstat_gwb = False
orf = 'hd'
crn_psd = 'powerlaw'
hyp = False
psd = 'powerlaw'
rn_psd = 'powerlaw'
dm_psd = 'powerlaw'
order = 1
crn_components = 10

N = int(1e6)

# psrdir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/'+psd+'/'+str(order)

if gwb:
    psrs = [psrs]

for psr in psrs:

    if gwb:
        pta_nonstat = cgw_model.model_gw(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict, crn_components=crn_components, orf=orf, crn_psd=crn_psd, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=nonstat_gwb, order=order, spline=True, par_0=None, par_1=None)
    else:
        pta_nonstat = cgw_model.model_spa(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=nonstat_gwb, order=order, spline=True)
    if hyp:
        if gwb:
            pta_stat = cgw_model.model_gw(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict, crn_components=crn_components, orf=orf, crn_psd=crn_psd, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=False, order=order)
        else:
            pta_stat = cgw_model.model_spa(psr, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict, rn_psd=rn_psd, dm_psd=dm_psd, nonstat=False, order=order)
        ptas = {}
        ptas[0] = pta_nonstat
        ptas[1] = pta_stat
        pta = HyperModel(ptas)
    else:
        pta = pta_nonstat
    print(pta.param_names)
    ndim = len(pta.param_names)

    x = np.hstack([p.sample() for p in pta_nonstat.params])
    # x = pta.initial_sample()
    # x[-3] = 5.
    # print(len(x))
    # print(len(pta.param_names))

    if hyp:
        x = np.append(x, np.random.uniform(-0.5, 1.5))
    # x = np.array([noisedict[p] for p in pta.param_names])

    # savedir = '/work/falxa/EPTA/nonstat/injections_normal/'+str(ninject) + '/'
    # savedir = './injections/'+str(ninject) + '/'
    if gwb:
        savedir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/gwb_slice_FINAL/'+str(alpha)+'/'
        # savedir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/ecc_ET_2nHz_wn_gwb/nohyp_'+str(psd)+'/'
        outdir = savedir + str(orf)
    else:
        savedir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/'+psd+'/'+str(order)+'/'+str(psr.name) + '_expdip/'
        # savedir = '/work/falxa/EPTA/nonstat/real_data_pta/'+pickle_name+'/ecc_ET_2nHz_wn_gwb/nohyp_'+str(psd)+'/'
        outdir = savedir + psr.name
    os.makedirs(outdir, exist_ok=True)
    ndim = len(pta.param_names)

    if hyp:
        sampler = pta.setup_sampler(outdir=outdir, resume=False, sample_nmodel=True, groups=None,
                        empirical_distr='/home/falxa/noises/emp/cw_search_-8.6_-8.0_patch_None_custom_models_ncgw_0_newsys_trim.pkl'
                         )
    else:
        sampler =  setup_sampler(pta, outdir=outdir, resume=False,
                        empirical_distr='/home/falxa/noises/emp/cw_search_-8.6_-8.0_patch_None_custom_models_ncgw_0_newsys_trim.pkl',
                        groups=None, human=None,
                        save_ext_dists=False, loglkwargs={}, logpkwargs={})

    # sampler for N steps
    sampler.sample(x, N, SCAMweight=50, AMweight=0, DEweight=50)

    if not gwb:
        chains = np.genfromtxt(outdir+'/chain_1.txt')
        burn = int(0.1*len(chains))
        chains = chains[burn:]
        nmodel = chains[:, -5]
        chains0 = chains[nmodel < 0.5]
        chains1 = chains[nmodel > 0.5]
        try:
            bf = len(chains0)/len(chains1)
            corner.corner(chains0[:, :-5], labels=np.array(pta.param_names)[:-1])
            plt.title('BF(nonstat/stat) ='+str(round(bf, 2)))
            plt.savefig(savedir + '/' + psr.name+'_nonstat.png')
            plt.cla()
            plt.clf()
            corner.corner(chains1[:, :-5], labels=np.array(pta.param_names)[:-1])
            plt.title('BF(nonstat/stat) ='+str(round(bf, 2)))
            plt.savefig(savedir + '/' + psr.name+'_stat.png')
            plt.cla()
            plt.clf()
        except:
            try:
                bf = len(chains0)/len(chains1)
                corner.corner(chains1[:, :-5], labels=np.array(pta.param_names)[:-1])
                plt.title('BF(nonstat/stat) ='+str(round(bf, 2)))
                plt.savefig(savedir + '/' + psr.name+'_stat.png')
                plt.cla()
                plt.clf()
            except:
                print('chibrage')
        plt.cla()
        plt.clf()