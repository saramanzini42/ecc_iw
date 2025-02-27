from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.stats
from collections import OrderedDict

import enterprise
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import white_signals
# from enterprise.signals import gp_signals
import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const
from enterprise.signals.selections import Selection
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp

from enterprise_extensions import model_utils
from enterprise_extensions import blocks
# import blocks
import nonstat_blocks
from enterprise_extensions import deterministic
from enterprise_extensions.chromatic import chromatic as chrom
from enterprise_extensions import model_orfs
from enterprise.signals.parameter import function

def dm_exponential_dip_fixed(idx=2, sign='negative', name='dmexp'):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        set sign of dip: 'positive', 'negative', or 'vary'
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dmexp = parameter.Constant()
    log10_Amp_dmexp = parameter.Constant()
    log10_tau_dmexp = parameter.Constant()
    if sign == 'vary':
        sign_param = parameter.Uniform(-1.0, 1.0)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0
    wf = chrom.chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                         t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                         sign_param=sign_param, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp

@function
def createfourierdesignmatrix_red(
    toas, nmodes=30, Tspan=None, dfs=None, logf=False, fmin=None, fmax=None, pshift=False, modes=None, pseed=None, fmargin=False
):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift
    :param pseed: option to provide phase shift seed
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    if fmargin:
        delta_f = np.random.uniform(0., 1/T)
        f += delta_f

    if dfs is not None:
        f += dfs

    # if requested, add random phase shift to basis functions
    if pshift or pseed is not None:
        if pseed is not None:
            # use the first toa to make a different seed for every pulsar
            seed = int(toas[0] / 17) + int(pseed)
            np.random.seed(seed)

        ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes)
    else:
        ranphase = np.zeros(nmodes)

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:, ::2] = np.sin(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])

    return F, Ffreqs

def FourierBasisCommonGP(
    spectrum,
    orf,
    coefficients=False,
    combine=True,
    logf=False,
    components=20,
    Tspan=None,
    modes=None,
    name="common_fourier",
    pshift=False,
    pseed=None,
    fmargin=False,
    dfs=None
):

    if coefficients and Tspan is None:
        raise ValueError(
            "With coefficients=True, FourierBasisCommonGP " + "requires that you specify Tspan explicitly."
        )


    basis = createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan, modes=modes, logf=logf, pshift=pshift, pseed=pseed, fmargin=fmargin, dfs=dfs)
    BaseClass = gp_signals.BasisCommonGP(spectrum, basis, orf, coefficients=coefficients, combine=combine, name=name)

    class FourierBasisCommonGP(BaseClass):
        signal_type = "common basis"
        signal_name = "common red noise"
        signal_id = name

        _Tmin, _Tmax = [], []

        def __init__(self, psr):
            super(FourierBasisCommonGP, self).__init__(psr)

            if Tspan is None:
                FourierBasisCommonGP._Tmin.append(psr.toas.min())
                FourierBasisCommonGP._Tmax.append(psr.toas.max())

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        # @signal_base.cache_call("basis_params", 1)
        def _construct_basis(self, params={}):
            span = Tspan if Tspan is not None else max(FourierBasisCommonGP._Tmax) - min(FourierBasisCommonGP._Tmin)
            self._basis, self._labels = self._bases(params=params, Tspan=span)

    return FourierBasisCommonGP

def FourierBasisGP(
    spectrum,
    coefficients=False,
    combine=True,
    logf=False,
    components=20,
    selection=Selection(selections.no_selection),
    Tspan=None,
    modes=None,
    name="red_noise",
    pshift=False,
    pseed=None,
    fmargin=False,
    dfs=None
):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan, modes=modes, logf=logf, pshift=pshift, pseed=pseed, fmargin=fmargin, dfs=dfs)
    BaseClass = gp_signals.BasisGP(spectrum, basis, coefficients=coefficients, combine=combine, selection=selection, name=name)

    class FourierBasisGP(BaseClass):
        signal_type = "basis"
        signal_name = "red noise"
        signal_id = name

    return FourierBasisGP


@function
def powerlaw_t(f, Nt=5, log10_A=-16., gamma=5., beta_amp=0., beta_gamma=0., components=2):
    df = np.diff(np.concatenate((np.array([0]), f[::components])))[::Nt]
    nmodes = int(len(f)/Nt)
    df = np.ones(len(f)) * df[1]
    # mmt = np.tile(np.arange(1, nmodes+1), Nt)
    nnt = np.repeat(np.arange(Nt), nmodes)
    log10_A_t = log10_A + beta_amp * nnt / Nt
    gamma_t = gamma + beta_gamma * nnt / Nt
    plaw = (10**log10_A_t) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma_t - 3) * f ** (-gamma_t) * df
    return (
        plaw
    )

@function
def free_spectrum_t(f, Nt=5, log10_rho=None, beta=None, Tspan=None):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return np.repeat(10 ** (2 * np.array(log10_rho)), 2)

@function
def powerlaw_bump(f, log10_A=-16, gamma=5, nf_bump=4, log10_amp_bump=-15, components=2):
    nbump = int(np.rint(nf_bump))
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    plaw = (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, components)
    plaw[2*nbump] += (10**log10_amp_bump)**2 * df[0] / 12.0 / np.pi**2 * const.fyr ** (-3)
    plaw[2*nbump+1] += (10**log10_amp_bump)**2 * df[0] / 12.0 / np.pi**2 * const.fyr ** (-3)
    return (
        plaw
    )

@function
def dTspan(Tspan, vary_Tspan):
    return Tspan + Tspan * vary_Tspan

def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, components=30, combine=True, logf=False, fmargin=False, fvary=False,
                           log10_A_val=None, gamma_val=None, delta_val=None, nf_bump=None,
                           logmin=None, logmax=None, anis_basis=None, lmax=0, psrs_pos=None, clm=None,
                           orf=None, orf_ifreq=0, leg_lmax=5, vary_Tspan=True,
                           name='gw', coefficients=False,
                           pshift=False, pseed=None):


    # Tspan = dTspan(Tspan, parameter.Uniform(-0.5, 0.5))

    if orf=='anis_orf' and clm is None:
        clm_name = '{}_clm'.format(name)
        if lmax > 0:
            clm = parameter.Normal(0., 1., size=(lmax+1)**2 - 1)('gw_clm')
        else:
            clm = np.zeros(2)
        # parameter.Uniform(-5., 5., size=(lmax+1)**2 - 1)('gw_clm')
    orfs = {'crn': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'anis_orf': model_orfs.anis_orf(params=clm, **{'anis_basis':anis_basis, 'lmax':lmax, 'psrs_pos':psrs_pos}),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_hd': model_orfs.param_hd_orf(a=parameter.Uniform(-1.5, 3.0)('gw_orf_param0'),
                                                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param1'),
                                                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param2')),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(-0.9, 0.9, size=7)('gw_orf_spline')),
            'bin_orf': model_orfs.bin_orf(params=parameter.Uniform(-1.0, 1.0, size=7)('gw_orf_bin')),
            'freq_hd': model_orfs.freq_hd(params=[components, orf_ifreq]),
            'legendre_orf': model_orfs.legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre'))}

    # common red noise parameters
    if psd in ['powerlaw', 'powerlaw_bump', 'turnover', 'turnover_knee', 'broken_powerlaw']:
        amp_name = '{}_log10_A'.format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)

        if logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            else:
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)

        else:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(-18, -14)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'powerlaw_bump':
            log10_Abump = parameter.Uniform(-14, -8)('{}_log10_A_bump'.format(name))
            if nf_bump is None:
                nf_bump = parameter.Uniform(0, components-1)('{}_nf_bump'.format(name))
            else:
                nf_bump = parameter.Constant(nf_bump)('{}_nf_bump'.format(name))
            cpl = powerlaw_bump(log10_A=log10_Agw, gamma=gamma_gw, log10_amp_bump=log10_Abump, nf_bump=nf_bump)
        elif psd == 'broken_powerlaw':
            delta_name = '{}_delta'.format(name)
            kappa_name = '{}_kappa'.format(name)
            log10_fb_name = '{}_log10_fb'.format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -7)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gpp.broken_powerlaw(log10_A=log10_Agw,
                                      gamma=gamma_gw,
                                      delta=delta_gw,
                                      log10_fb=log10_fb_gw,
                                      kappa=kappa_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gpp.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                    lfb=lfb_gw, lfk=lfk_gw,
                                    kappa=kappa_gw, delta=delta_gw)

    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)

        # checking if priors specified, otherwise give default values
        if logmin is None:
            logmin = -9
        if logmax is None:
            logmax = -4

        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(logmin, logmax,
                                               size=components)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if fvary:
        # dfs = parameter.Uniform(0., 1/Tspan, size=components)('df')
        dfs = parameter.Uniform(0., 1/Tspan)('df')
    else:
        dfs = None

    if orf is None:
        crn = FourierBasisGP(cpl, coefficients=coefficients, logf=logf, fmargin=fmargin,
                                        components=components, Tspan=Tspan, dfs=dfs,
                                        name=name, pshift=pshift, pseed=pseed)
    elif orf in orfs.keys():
        if orf == 'crn':
            crn = FourierBasisGP(cpl, coefficients=coefficients, fmargin=fmargin,
                                            components=components, Tspan=Tspan,
                                            name=name, pshift=pshift, pseed=pseed)
        else:
            crn = FourierBasisCommonGP(cpl, orfs[orf], logf=logf, fmargin=fmargin,
                                                  components=components, dfs=dfs,
                                                  Tspan=Tspan,
                                                  name=name, pshift=pshift,
                                                  pseed=pseed)
    elif isinstance(orf, types.FunctionType):
        crn = FourierBasisCommonGP(cpl, orf, logf=logf, fmargin=fmargin,
                                              components=components, dfs=dfs,
                                              Tspan=Tspan,
                                              name=name, pshift=pshift,
                                              pseed=pseed)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn

@signal_base.function
def cw_delay(toas, pos, pdist,
             cos_gwtheta=0, gwphi=0, cos_inc=0,
             log10_mc=9, log10_fgw=-8, log10_dist=None, log10_h=None,
             phase0=0, psi=0, k_drop=1.,
             psrTerm=False, p_dist=1, p_phase=None,
             evolve=False, phase_approx=False, check=False,
             tref=0):
    """
    Function to create GW incuced residuals from a SMBMB as
    defined in Ellis et. al 2012,2013.
    :param toas:
        Pular toas in seconds
    :param pos:
        Unit vector from the Earth to the pulsar
    :param pdist:
        Pulsar distance (mean and uncertainty) [kpc]
    :param cos_gwtheta:
        Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi:
        Azimuthal angle of GW source in celestial coords [radians]
    :param cos_inc:
        cosine of Inclination of GW source [radians]
    :param log10_mc:
        log10 of Chirp mass of SMBMB [solar masses]
    :param log10_fgw:
        log10 of Frequency of GW (twice the orbital frequency) [Hz]
    :param log10_dist:
        log10 of Luminosity distance to SMBMB [Mpc],
        used to compute strain, if not None
    :param log10_h:
        log10 of GW strain,
        used to compute distance, if not None
    :param phase0:
        Initial GW phase of source [radians]
    :param psi:
        Polarization angle of GW source [radians]
    :param psrTerm:
        Option to include pulsar term [boolean]
    :param p_dist:
        Pulsar distance parameter
    :param p_phase:
        Use pulsar phase to determine distance [radian]
    :param evolve:
        Option to include/exclude full evolution [boolean]
    :param phase_approx:
        Option to include/exclude phase evolution across observation time
        [boolean]
    :param check:
        Check if frequency evolves significantly over obs. time [boolean]
    :param tref:
        Reference time for phase and frequency [s]
    :return: Vector of induced residuals
    """

    # convert units to time
    mc = 10**log10_mc * const.Tsun
    fgw = 10**log10_fgw
    gwtheta = np.arccos(cos_gwtheta)
    inc = np.arccos(cos_inc)
    p_dist = (pdist[0] + pdist[1]*p_dist)*const.kpc/const.c

    if log10_h is None and log10_dist is None:
        raise ValueError("one of log10_dist or log10_h must be non-None")
    elif log10_h is not None and log10_dist is not None:
        raise ValueError("only one of log10_dist or log10_h can be non-None")
    elif log10_h is None:
        dist = 10**log10_dist * const.Mpc / const.c
    else:
        dist = 2 * mc**(5/3) * (np.pi*fgw)**(2/3) / 10**log10_h

    if check:
        # check that frequency is not evolving significantly over obs. time
        fstart = fgw * (1 - 256/5 * mc**(5/3) * fgw**(8/3) * toas[0])**(-3/8)
        fend = fgw * (1 - 256/5 * mc**(5/3) * fgw**(8/3) * toas[-1])**(-3/8)
        df = fend - fstart

        # observation time
        Tobs = toas.max()-toas.min()
        fbin = 1/Tobs

        if np.abs(df) > fbin:
            print('WARNING: Frequency is evolving over more than one '
                  'frequency bin.')
            print('f0 = {0}, f1 = {1}, df = {2}, fbin = {3}'.format(fstart, fend, df, fbin))
            return np.ones(len(toas)) * np.nan

    # get antenna pattern funcs and cosMu
    # write function to get pos from theta,phi
    fplus, fcross, cosMu = utils.create_gw_antenna_pattern(pos, gwtheta, gwphi)

    # get pulsar time
    toas -= tref
    if p_dist > 0:
        tp = toas-p_dist*(1-cosMu)
    else:
        tp = toas

    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2  # convert GW to orbital phase
    # omegadot = 96/5 * mc**(5/3) * w0**(11/3) # Not currently used in code

    # evolution
    if evolve:
        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)

        if p_dist > 0:
            omega_p0 = w0 * (1 + 256/5
                             * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)
        else:
            omega_p0 = w0

        # calculate time dependent phase
        phase = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))

        if p_phase is None:
            phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3))
        else:
            phase_p = (phase0 + p_phase
                       + 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3)))

    elif phase_approx:
        # monochromatic
        omega = w0
        if p_dist > 0:
            omega_p = w0 * (1 + 256/5
                            * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)
        else:
            omega_p = w0

        # phases
        phase = phase0 + omega * toas
        if p_phase is not None:
            phase_p = phase0 + p_phase + omega_p*toas
        else:
            phase_p = (phase0 + omega_p*toas
                       + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3)))

    # no evolution
    else:
        # monochromatic
        omega = np.pi*fgw
        omega_p = omega

        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp

    # define time dependent coefficients
    At = -0.5*np.sin(2*phase)*(3+np.cos(2*inc))
    Bt = 2*np.cos(2*phase)*np.cos(inc)
    At_p = -0.5*np.sin(2*phase_p)*(3+np.cos(2*inc))
    Bt_p = 2*np.cos(2*phase_p)*np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))

    # define rplus and rcross
    rplus = alpha*(-At*np.cos(2*psi)+Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(-At_p*np.cos(2*psi)+Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))

    # k dropout
    k = np.rint(k_drop)

    # residuals
    if psrTerm:
        res = k*fplus*(rplus_p-rplus) + k*fcross*(rcross_p-rcross)
    else:
        res = -k*fplus*rplus - k*fcross*rcross

    return res


@signal_base.function
def compute_eccentric_residuals(toas, theta, phi, cos_gwtheta, gwphi,
                                log10_mc, log10_dist, log10_h, log10_F, cos_inc,
                                psi, gamma0, e0, l0, q, nmax=400, pdist=1.0,
                                pphase=None, pgam=None, psrTerm=False,
                                tref=0, check=False):
    """
    Simulate GW from eccentric SMBHB. Waveform models from
    Taylor et al. (2015) and Barack and Cutler (2004).
    WARNING: This residual waveform is only accurate if the
    GW frequency is not significantly evolving over the
    observation time of the pulsar.
    :param toa: pulsar observation times
    :param theta: polar coordinate of pulsar
    :param phi: azimuthal coordinate of pulsar
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param log10_mc: Base-10 lof of chirp mass of SMBMB [solar masses]
    :param log10_dist: Base-10 uminosity distance to SMBMB [Mpc]
    :param log10_F: base-10 orbital frequency of SMBHB [Hz]
    :param inc: Inclination of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param gamma0: Initial angle of periastron [radians]
    :param e0: Initial eccentricity of SMBHB
    :param l0: Initial mean anomoly [radians]
    :param q: Mass ratio of SMBHB
    :param nmax: Number of harmonics to use in waveform decomposition
    :param pdist: Pulsar distance [kpc]
    :param pphase: Pulsar phase [rad]
    :param pgam: Pulsar angle of periastron [rad]
    :param psrTerm: Option to include pulsar term [boolean]
    :param tref: Fidicuial time at which initial parameters are referenced [s]
    :param check: Check if frequency evolves significantly over obs. time
    :returns: Vector of induced residuals
    """

    # convert from sampling
    F = 10.0**log10_F
    mc = 10.0**log10_mc
    dist = 10.0**log10_dist
    if log10_h is not None:
        h0 = 10.0**log10_h
    else:
        h0 = None
    inc = np.arccos(cos_inc)
    gwtheta = np.arccos(cos_gwtheta)

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    # pulsar position vector
    phat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                     np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    # get values from pulsar object
    toas = toas.copy() - tref

    if check:
        # check that frequency is not evolving significantly over obs. time
        y = utils.solve_coupled_ecc_solution(F, e0, gamma0, l0, mc, q,
                                             np.array([0.0, toas.max()]))

        # initial and final values over observation time
        Fc0, ec0, gc0, phic0 = y[0, :]
        Fc1, ec1, gc1, phic1 = y[-1, :]

        # observation time
        Tobs = 1/(toas.max()-toas.min())

        if np.abs(Fc0-Fc1) > 1/Tobs:
            print('WARNING: Frequency is evolving over more than one frequency bin.')
            print('F0 = {0}, F1 = {1}, delta f = {2}'.format(Fc0, Fc1, 1/Tobs))
            return np.ones(len(toas)) * np.nan

    # get gammadot for earth term
    gammadot = utils.get_gammadot(F, mc, q, e0)

    # get number of harmonics to use
    if not isinstance(nmax, int):
        if isinstance(nmax, str):
            f1yr = 1/(3600*24*365.25)
            nharm = int(f1yr/F)
        elif e0 < 0.999 and e0 > 0.001:
            nharm = int(nmax(e0))
        elif e0 < 0.001:
            nharm = 2
        else:
            nharm = int(nmax(0.999))
    else:
        nharm = nmax

    # no more than 100 harmonics
    nharm = min(nharm, 100)

    ##### earth term #####
    splus, scross = utils.calculate_splus_scross(nmax=nharm, mc=mc, dl=dist,
                                                 h0=h0, F=F, e=e0, t=toas.copy(),
                                                 l0=l0, gamma=gamma0,
                                                 gammadot=gammadot, inc=inc)

    ##### pulsar term #####
    if psrTerm:
        # pulsar distance
        pd = pdist

        # convert units
        pd *= const.kpc / const.c

        # get pulsar time
        tp = toas.copy() - pd * (1-cosMu)

        # solve coupled system of equations to get pulsar term values
        y = utils.solve_coupled_ecc_solution(F, e0, gamma0, l0, mc,
                                             q, np.array([0.0, tp.min()]))

        # get pulsar term values
        if np.any(y):
            Fp, ep, gp, phip = y[-1, :]

            # get gammadot at pulsar term
            gammadotp = utils.get_gammadot(Fp, mc, q, ep)

            # get phase at pulsar
            if pphase is None:
                lp = phip
            else:
                lp = pphase

            # get angle of periastron at pulsar
            if pgam is None:
                gp = gp
            else:
                gp = pgam

            # get number of harmonics to use
            if not isinstance(nmax, int):
                if e0 < 0.999 and e0 > 0.001:
                    nharm = int(nmax(e0))
                elif e0 < 0.001:
                    nharm = 2
                else:
                    nharm = int(nmax(0.999))
            else:
                nharm = nmax

            # no more than 1000 harmonics
            nharm = min(nharm, 100)
            splusp, scrossp = utils.calculate_splus_scross(nmax=nharm, mc=mc,
                                                           dl=dist, h0=h0,
                                                           F=Fp, e=ep,
                                                           t=toas.copy(),
                                                           l0=lp, gamma=gp,
                                                           gammadot=gammadotp,
                                                           inc=inc)

            rr = (fplus*cos2psi - fcross*sin2psi) * (splusp - splus) + \
                (fplus*sin2psi + fcross*cos2psi) * (scrossp - scross)

        else:
            rr = np.ones(len(toas)) * np.nan

    else:
        rr = - (fplus*cos2psi - fcross*sin2psi) * splus - \
            (fplus*sin2psi + fcross*cos2psi) * scross

    return rr

# Extra model components not part of base enterprise ####
def cw_block_circ(amp_prior='log-uniform', dist_prior=None, drop=False, drop_psr=False,
                  skyloc=None, log10_fgw=None, patch=None,
                  psrTerm=False, tref=0, name='cw'):
    """
    Returns deterministic, cirular orbit continuous GW model:
    :param amp_prior:
        Prior on log10_h. Default is "log-uniform."
        Use "uniform" for upper limits, or "None" to search over
        log10_dist instead.
    :param dist_prior:
        Prior on log10_dist. Default is "None," meaning that the
        search is over log10_h instead of log10_dist. Use "log-uniform"
        to search over log10_h with a log-uniform prior.
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_fgw:
        Fixed log10 GW frequency of CW signal search.
        Search over GW frequency if ``None`` given.
    :param patch:
        Bounds for sky position prior [[costh_min, costh_max], [phi_min, phi_max]]
    :param ecc:
        Fixed log10 distance to SMBHB search.
        Search over distance or strain if ``None`` given.
    :param psrTerm:
        Boolean for whether to include the pulsar term. Default is False.
    :param name:
        Name of CW signal.
    """

    if dist_prior == None:
        log10_dist = None

        if amp_prior == 'uniform':
            log10_h = parameter.LinearExp(-16.0, -11.0)('{}_log10_h'.format(name))
        elif amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(-16.0, -11.0)('{}_log10_h'.format(name))

    elif dist_prior == 'log-uniform':
        log10_dist = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))
        log10_h = None

    # chirp mass [Msol]
    if psrTerm:
        log10_Mc = parameter.Uniform(7.0, 11.0)('{}_log10_Mc'.format(name))
    else:
        log10_Mc = parameter.Constant(9.0)('{}_log10_Mc'.format(name))

    # GW frequency [Hz]
    if log10_fgw is None:
        log10_fgw = parameter.Uniform(-9.0, -7.0)('{}_log10_fgw'.format(name))
    elif isinstance(log10_fgw, list):
        log10_fgw = parameter.Uniform(log10_fgw[0], log10_fgw[1])('{}_log10_fgw'.format(name))
    else:
        log10_fgw = parameter.Constant(log10_fgw)('{}_log10_fgw'.format(name))
    # orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    # initial GW phase [radians]
    phase0 = parameter.Uniform(0.0, 2*np.pi)('{}_phase0'.format(name))

    # polarization
    psi_name = '{}_psi'.format(name)
    psi = parameter.Uniform(0, np.pi)(psi_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        if patch is None:
            costh = parameter.Uniform(-1, 1)(costh_name)
            phi = parameter.Uniform(0, 2*np.pi)(phi_name)
        else:
            costh = parameter.Uniform(patch[0, 0], patch[0, 1])(costh_name)
            phi = parameter.Uniform(patch[1, 0], patch[1, 1])(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    if psrTerm:
        p_phase = parameter.Uniform(0, 2*np.pi)
        p_dist = parameter.Normal(0, 1)
    else:
        p_phase = None
        p_dist = 0

    if drop:
        if drop_psr:
            k_drop = parameter.Uniform(0, 1)
        else:
            kdrop_name = '{}_k_drop'.format(name)
            k_drop = parameter.Uniform(0, 1)(kdrop_name)
    else:
        k_drop = 1.

    # continuous wave signal
    wf = cw_delay(cos_gwtheta=costh, gwphi=phi, cos_inc=cosinc,
                  log10_mc=log10_Mc, log10_fgw=log10_fgw,
                  log10_h=log10_h, log10_dist=log10_dist,
                  phase0=phase0, psi=psi, k_drop=k_drop,
                  psrTerm=psrTerm, p_dist=p_dist, p_phase=p_phase,
                  phase_approx=False, check=False,
                  tref=tref)
    cw = deterministic.CWSignal(wf, ecc=False, psrTerm=psrTerm)

    return cw

def cw_block_ecc(amp_prior='log-uniform', skyloc=None, log10_F=None, patch=None, nmax='1yr',
                 ecc=None, psrTerm=False, tref=0, name='cw'):
    """
    Returns deterministic, eccentric orbit continuous GW model:
    :param amp_prior:
        Prior on log10_h and log10_Mc/log10_dL. Default is "log-uniform" with
        log10_Mc and log10_dL searched over. Use "uniform" for upper limits,
        log10_h searched over.
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_F:
        Fixed log-10 orbital frequency of CW signal search.
        Search over orbital frequency if ``None`` given.
    :param patch:
        Bounds for sky position prior [[costh_min, costh_max], [phi_min, phi_max]]
    :param ecc:
        Fixed eccentricity of SMBHB search.
        Search over eccentricity if ``None`` given.
    :param psrTerm:
        Boolean for whether to include the pulsar term. Default is False.
    :param name:
        Name of CW signal.
    """

    if amp_prior == 'uniform':
        log10_h = parameter.LinearExp(-18.0, -11.0)('{}_log10_h'.format(name))
    elif amp_prior == 'log-uniform':
        log10_h = parameter.Uniform(-18.0, -11.0)('{}_log10_h'.format(name))
    # chirp mass [Msol]
    log10_Mc = parameter.Uniform(6.0, 10.0)('{}_log10_Mc'.format(name))
    # luminosity distance [Mpc]
    log10_dL = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))

    # orbital frequency [Hz]
    if log10_F is None:
        log10_Forb = parameter.Uniform(-9.0, -7.0)('{}_log10_Forb'.format(name))
    elif isinstance(log10_F, list):
        log10_Forb = parameter.Uniform(log10_F[0], log10_F[1])('{}_log10_Forb'.format(name))
    else:
        log10_Forb = parameter.Constant(log10_F)('{}_log10_Forb'.format(name))
    # orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    # periapsis position angle [radians]
    gamma_0 = parameter.Uniform(0.0, np.pi)('{}_gamma0'.format(name))

    # Earth-term eccentricity
    if ecc is None:
        e_0 = parameter.Uniform(0.0, 0.99)('{}_e0'.format(name))
    elif isinstance(ecc, list):
        e_0 = parameter.Uniform(ecc[0], ecc[1])('{}_e0'.format(name))
    else:
        e_0 = parameter.Constant(ecc)('{}_e0'.format(name))

    # initial mean anomaly [radians]
    l_0 = parameter.Uniform(0.0, 2.0*np.pi)('{}_l0'.format(name))
    # mass ratio = M_2/M_1
    q = parameter.Constant(1.0)('{}_q'.format(name))

    # polarization
    pol_name = '{}_pol'.format(name)
    pol = parameter.Uniform(0, np.pi)(pol_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        if patch is None:
            costh = parameter.Uniform(-1, 1)(costh_name)
            phi = parameter.Uniform(0, 2*np.pi)(phi_name)
        else:
            costh = parameter.Uniform(patch[0, 0], patch[0, 1])(costh_name)
            phi = parameter.Uniform(patch[1, 0], patch[1, 1])(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    # continuous wave signal
    wf = compute_eccentric_residuals(cos_gwtheta=costh, gwphi=phi,
                                     log10_mc=log10_Mc, log10_dist=log10_dL,
                                     log10_h=log10_h, log10_F=log10_Forb,
                                     cos_inc=cosinc, psi=pol, gamma0=gamma_0,
                                     e0=e_0, l0=l_0, q=q, nmax=nmax,
                                     pdist=None, pphase=None, pgam=None,
                                     tref=tref, check=False)
    cw = deterministic.CWSignal(wf, ecc=True, psrTerm=psrTerm)

    return cw


def select_new_backends(psr, backend_list):

    mask = [i for i, flag in enumerate(psr.backend_flags[psr._iisort]) if np.any([bl in flag for bl in backend_list])]
    psr._toas = psr._toas[mask]
    psr._toaerrs = psr._toaerrs[mask]
    psr._residuals = psr._residuals[mask]
    psr._ssbfreqs = psr._ssbfreqs[mask]

    psr._designmatrix = psr._designmatrix[mask, :]
    dmx_mask = np.sum(psr._designmatrix, axis=0) != 0.0
    psr._designmatrix = psr._designmatrix[:, dmx_mask]

    if isinstance(psr._flags, np.ndarray):
        psr._flags = psr._flags[mask]
    else:
        for key in psr._flags:
            psr._flags[key] = psr._flags[key][mask]

    if psr._planetssb is not None:
        psr._planetssb = psr.planetssb[mask, :, :]

    psr.sort_data()
    return psr


def model_cw(psrs, upper_limit=False, prefix='cw', custom_models={}, noisedict=None, n_cgw=1, orf=None, drop=False, drop_psr=False, crn_logf=False, fvary_crn=False,
             rn_var=True, rn_psd='powerlaw', dm_var=True, dm_psd='powerlaw', components=30, dm_components=None, J1713_expdip=True, crn_name='gw',
             crn=False, bayesephem=False, skyloc=None, patch=None, log10_F=None, ecc=False, crn_psd='powerlaw', crn_components=30, crn_Tspan=None,
             psrTerm=False, wideband=False, anis_basis=None, lmax=0, psrs_pos=None, clm=None, nf_bump=None, fmargin_crn=False, fmargin_noise=False):
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    # timing model
    s0 = gp_signals.TimingModel(use_svd=True)

    # ephemeris model
    if bayesephem:
        s0 += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    if crn:
        if crn_Tspan is None:
            crn_Tspan = Tspan
        vary_Tspan = False
        s0 += blocks.common_red_noise_block(psd=crn_psd, prior='log-uniform', Tspan=crn_Tspan, components=crn_components, orf=orf, name=crn_name)

        # s0 += common_red_noise_block(psd=crn_psd, prior='log-uniform', Tspan=crn_Tspan, logf=crn_logf, components=crn_components, orf=orf, name=crn_name,
        #                                     anis_basis=anis_basis, lmax=lmax, psrs_pos=psrs_pos, clm=clm, nf_bump=nf_bump, fmargin=fmargin_crn, fvary=fvary_crn)
    # GW CW signal block
    prefixes = [prefix]
    for ngw in range(n_cgw-1):
        prefixes.append(prefix+'_'+str(ngw+1))
    for ngw in range(n_cgw):
        if not ecc:
            s0 += cw_block_circ(amp_prior=amp_prior, patch=patch,
                    skyloc=skyloc, log10_fgw=log10_F, drop=drop, drop_psr=drop_psr,
                    psrTerm=psrTerm, tref=tmin, name=prefixes[ngw])
        else:
            if type(ecc) is not float:
                e0 = None
            if isinstance(ecc, list):
                e0 = ecc
            s0 += cw_block_ecc(amp_prior=amp_prior, patch=patch,
                            skyloc=skyloc, log10_F=log10_F, ecc=e0,
                            psrTerm=psrTerm, tref=tmin, name=prefixes[ngw])

    models = []
    for p in psrs:

        # adding white-noise, red noise, dm noise, chromatic noise and acting on psr objects
        # white noise
        s3 = s0 + blocks.white_noise_block(vary=False, inc_ecorr=False, tnequad=True, name='')
        if p.name in [*custom_models]:

            rn_components = custom_models[p.name]['RN']
            dm_components = custom_models[p.name]['DM']
            sv_components = custom_models[p.name]['Sv']

            # print(rn_components, dm_components, sv_components)

            # red noise
            if rn_components is not None:
                s3 += blocks.red_noise_block(prior=amp_prior, psd=rn_psd, Tspan=None, components=rn_components)
            # dm noise
            if dm_components is not None:
                s3 += blocks.dm_noise_block(psd=dm_psd, prior=amp_prior, components=dm_components, Tspan=None, gamma_val=None)
            # scattering variation
            if sv_components is not None:
                s3 += blocks.chromatic_noise_block(gp_kernel='diag', psd=rn_psd, prior=amp_prior, idx=4, Tspan=None, components=sv_components)
            if p.name == 'J1713+0747' and J1713_expdip:
                s3 += chrom.dm_exponential_dip(tmin=54650, tmax=54850, idx=4, sign='negative', name='expd-%s_%s_%s'%(4, int(54650),int(54850)))
                s3 += chrom.dm_exponential_dip(tmin=57490, tmax=57530, idx=1, sign='negative', name='expd-%s_%s_%s'%(1, int(57490),int(57530)))
            
        models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta

def model_gw(psrs, upper_limit=False, J1713_nonstat=False, J1713_expdip=False, prefix='cw', custom_models={}, noisedict=None, orf=None, crn_components=30, par_0=None, par_1=None,
            rn_psd='powerlaw', dm_psd='powerlaw', crn_name='gw', crn_psd='powerlaw', nonstat=True, order=1, logf=False, sample_knot_t=False, spline=False, crn_Tspan=None):

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.amin([psr.toas.min() for psr in psrs])
    tmax = np.amax([psr.toas.max() for psr in psrs])
    Tspan = tmax - tmin
    if crn_Tspan is None:
        crn_Tspan = Tspan

    # timing model
    s0 = gp_signals.TimingModel(use_svd=True)

    # gwb
    if nonstat:
        s0 += nonstat_blocks.common_time_correlated_block_t(psd=crn_psd+'_cos_t', name=crn_name, prior=amp_prior, Tspan=crn_Tspan, components=crn_components, orf=orf, order=order, par_0_val=par_0, par_1_val=par_1, idx=0, logf=logf, spline=spline, sample_knot_t=sample_knot_t)
    else:
        s0 += blocks.common_red_noise_block(psd=crn_psd, prior='log-uniform', Tspan=crn_Tspan, components=crn_components, orf=orf, name=crn_name)

    models = []
    for p in psrs:

        # white noise
        s3 = s0 + blocks.white_noise_block(vary=False, inc_ecorr=False, tnequad=True, name='')
        if p.name in [*custom_models]:

            rn_components = custom_models[p.name]['RN']
            dm_components = custom_models[p.name]['DM']
            sv_components = custom_models[p.name]['Sv']

            if ('J1713' in p.name) and J1713_nonstat:
                if rn_components is not None:
                    s3 += blocks.red_noise_block(prior=amp_prior, psd=rn_psd, Tspan=Tspan, components=rn_components)
                # if rn_components is not None:
                #     s3 += nonstat_blocks.time_correlated_block_t(psd='powerlaw_spline_t', name='red_noise', prior=amp_prior, Tspan=Tspan, components=rn_components, order=order, idx=0, logf=logf, sample_knot_t=sample_knot_t)
                # # dm noise
                if dm_components is not None:
                    s3 += nonstat_blocks.time_correlated_block_t(psd='powerlaw_spline_t', name='dm_gp', prior=amp_prior, components=dm_components, Tspan=Tspan, order=order, idx=2, logf=logf, sample_knot_t=sample_knot_t, par_0_val=None, par_1_val=None)
                # scattering variation
                if sv_components is not None:
                    s3 += nonstat_blocks.time_correlated_block_t(psd='powerlaw_spline_t', name='chrom_gp', prior=amp_prior, components=sv_components, Tspan=Tspan, order=order, idx=4, logf=logf, sample_knot_t=sample_knot_t)
            else:
                # red noise
                if rn_components is not None:
                    s3 += blocks.red_noise_block(prior=amp_prior, psd=rn_psd, components=rn_components)
                # dm noise
                if dm_components is not None:
                    s3 += blocks.dm_noise_block(psd=dm_psd, prior=amp_prior, components=dm_components)
                # scattering variation
                if sv_components is not None:
                    s3 += blocks.chromatic_noise_block(gp_kernel='diag', psd=dm_psd, prior=amp_prior, idx=4, components=sv_components)
                if p.name == 'J1713+0747' and J1713_expdip:
                    s3 += dm_exponential_dip_fixed(idx=4, sign='negative', name='expd-%s_%s_%s'%(4, int(54650),int(54850)))
                    s3 += dm_exponential_dip_fixed(idx=1, sign='negative', name='expd-%s_%s_%s'%(1, int(57490),int(57530)))
                
        models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    expdips = {'J1713+0747_expd-1_57490_57530_log10_Amp' : -5.85,
            'J1713+0747_expd-1_57490_57530_log10_tau' : 1.4,
            'J1713+0747_expd-1_57490_57530_t0' : 57511.0,
            'J1713+0747_expd-4_54650_54850_log10_Amp' : -5.55,
            'J1713+0747_expd-4_54650_54850_log10_tau' : 1.55,
            'J1713+0747_expd-4_54650_54850_t0' : 54752.5}

    noisedict.update(expdips)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta

def model_spa(psr, upper_limit=False, vary_wn=False, J1713_expdip=False, prefix='cw', custom_models={}, noisedict=None, rn_psd='powerlaw', dm_psd='powerlaw', nonstat=True, order=1, logf=False, sample_knot_t=False, spline=False):

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = psr.toas.min()
    tmax = psr.toas.max()
    Tspan = tmax - tmin

    # timing model
    s0 = gp_signals.TimingModel(use_svd=True)

    models = []
    for p in [psr]:

        # white noise
        s3 = s0 + blocks.white_noise_block(vary=True, inc_ecorr=False, tnequad=True, name='')
        if p.name in [*custom_models]:

            rn_components = custom_models[p.name]['RN']
            dm_components = custom_models[p.name]['DM']
            sv_components = custom_models[p.name]['Sv']

            # red noise
            if nonstat:
                if spline:
                    rn_psd += '_spline'
                    dm_psd += '_spline'

                # if rn_components is not None:
                #     s3 += blocks.red_noise_block(prior=amp_prior, psd='powerlaw', Tspan=Tspan, components=rn_components)
                # # dm noise
                # if dm_components is not None:
                #     s3 += blocks.dm_noise_block(psd='powerlaw', prior=amp_prior, components=dm_components, Tspan=Tspan, gamma_val=None)
                if rn_components is not None:
                    s3 += nonstat_blocks.time_correlated_block_t(psd=rn_psd+'_t', name='red_noise', prior=amp_prior, Tspan=Tspan, components=rn_components, order=order, idx=0, logf=logf, sample_knot_t=sample_knot_t)
                # dm noise
                if dm_components is not None:
                    s3 += nonstat_blocks.time_correlated_block_t(psd=dm_psd+'_t', name='dm_gp', prior=amp_prior, components=dm_components, Tspan=Tspan, order=order, idx=2, logf=logf, sample_knot_t=sample_knot_t, par_0_val=None, par_1_val=None)
                # scattering variation
                if sv_components is not None:
                    s3 += nonstat_blocks.time_correlated_block_t(psd=dm_psd+'_t', name='chrom_gp', prior=amp_prior, components=sv_components, Tspan=Tspan, order=order, idx=4, logf=logf, sample_knot_t=sample_knot_t)
            else:
                if rn_components is not None:
                    s3 += blocks.red_noise_block(prior=amp_prior, psd=rn_psd, Tspan=Tspan, components=rn_components)
                # dm noise
                if dm_components is not None:
                    s3 += blocks.dm_noise_block(psd=dm_psd, prior=amp_prior, components=dm_components, Tspan=Tspan, gamma_val=None)
                # scattering variation
                if sv_components is not None:
                    s3 += blocks.chromatic_noise_block(gp_kernel='diag', psd=rn_psd, prior=amp_prior, idx=4, Tspan=Tspan, components=sv_components)

            if p.name == 'J1713+0747' and J1713_expdip:
                # s3 += dm_exponential_dip_fixed(idx=4, sign='negative', name='expd-%s_%s_%s'%(4, int(54650),int(54850)))
                # s3 += dm_exponential_dip_fixed(idx=1, sign='negative', name='expd-%s_%s_%s'%(1, int(57490),int(57530)))
                s3 += dm_exponential_dip_fixed(idx=6.4, sign='negative', name='expd-%s_%s_%s'%(6.4, int(57490),int(57530)))
                s3 += dm_exponential_dip_fixed(idx=0.7, sign='negative', name='expd-%s_%s_%s'%(0.7, int(59316),int(59400)))
            
        models.append(s3(p))

    # set up PTA
    pta = signal_base.PTA(models)

    expdips = {
            # 'J1713+0747_expd-1_57490_57530_log10_Amp' : -5.85,
            # 'J1713+0747_expd-1_57490_57530_log10_tau' : 1.4,
            # 'J1713+0747_expd-1_57490_57530_t0' : 57511.0,
            # 'J1713+0747_expd-4_54650_54850_log10_Amp' : -5.55,
            # 'J1713+0747_expd-4_54650_54850_log10_tau' : 1.55,
            # 'J1713+0747_expd-4_54650_54850_t0' : 54752.5,
            "J1713+0747_expd-6.4_57490_57530_idx": 6.427994318424217,
            "J1713+0747_expd-6.4_57490_57530_log10_Amp": -5.366493177923624,
            "J1713+0747_expd-6.4_57490_57530_log10_tau": 0.5009253741406431,
            "J1713+0747_expd-6.4_57490_57530_t0": 57511.81775868528,
            "J1713+0747_expd-0.7_59316_59400_idx": 0.7062121689539006,
            "J1713+0747_expd-0.7_59316_59400_log10_Amp": -4.518747342529928,
            "J1713+0747_expd-0.7_59316_59400_log10_tau": 2.0973903163613086,
            "J1713+0747_expd-0.7_59316_59400_t0": 59320.640378548065}

    noisedict.update(expdips)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta