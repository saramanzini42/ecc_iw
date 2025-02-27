import numpy as np
import os
import pickle
import json
import optparse
from enterprise_extensions.sampler import JumpProposal
from enterprise_extensions.sampler import setup_sampler
import cgw_model
# from enterprise_extensions.hypermodel import HyperModel

pickle_name = 'psrs_epta_dr2_1_3_de440'
noisedict = json.load(open('/home/falxa/noises/noisefile_epta_dr2_v1_3.json', 'rb'))
custom_models = json.load(open('/home/falxa/noises/custom_models_final.json'))

psrs = pickle.load(open('/work/falxa/EPTA/pkl/'+pickle_name+'.pkl', 'rb'))

# set noise parameters
gwb = False
orf = None
hyp = True
psd = 'powerlaw_spline'
rn_psd = 'powerlaw'
dm_psd = 'powerlaw'
order = 1

N = int(1e6)
# prepare pta
pta = cgw_model.model_gw(psrs, upper_limit=False, J1713_expdip=False, custom_models=custom_models, noisedict=noisedict,
                 orf='hd', rn_psd=rn_psd, dm_psd=dm_psd, nonstat=True, order=order, spline=True)

savedir = '/work/falxa/EPTA/nonstat/real_data_pta/psrs_epta_dr2_1_3_de440/gwb/1/None_1'

chains = np.genfromtxt(savedir+'/chain_1.txt')
pars = np.genfromtxt(savedir+'/pars.txt', dtype='str')

nmodel = chains[:, -5]
chains = chains[nmodel < 0.5]

burn = int(0.1*len(chains))
thin = 1
chains = chains[burn::thin]

outdir = savedir

log_weights = []
for s in chains:
    lnlike_new = pta.get_lnlikelihood(s[:-5])
    log_weights.append(lnlike_new - s[-3])
    print(log_weights[-1])
    np.savetxt(outdir+'/log_weights.txt', np.array(log_weights))

np.savetxt(outdir+'/log_weights.txt', np.array(log_weights))
