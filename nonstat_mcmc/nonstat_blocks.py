import enterprise
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import white_signals
# from enterprise.signals import gp_signals
import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const
from enterprise.signals import gp_priors as gpp

from enterprise_extensions import model_utils
from enterprise_extensions import blocks
from enterprise_extensions import deterministic
from enterprise_extensions.chromatic import chromatic as chrom
import matplotlib.pyplot as plt
from enterprise_extensions import model_orfs
from enterprise.signals.parameter import function
from enterprise.signals.gp_signals import BasisGP
import numpy as np
from scipy.special import betainc
import scipy.interpolate as interpolate


def g_chebyshev(a, toas, Tspan):

    t = (toas - toas[0]) / Tspan
    coeffs = np.append(0, a)
    g = np.polynomial.chebyshev.Chebyshev(coeffs, domain=[0, 1])
    return g(t)

def g_spline(knot_t, knot_g, order, toas, Tspan):

    times = (toas - toas[0]) / Tspan
    ts = np.linspace(0, 1, order+1)
    gts = np.append(0., knot_g)
    k = min(5, order)
    spline = interpolate.UnivariateSpline(ts, gts, k=k)
    return spline(times)

def exp_window(t0, width, toas, Tspan):

    t = (toas - toas[0]) / Tspan
    window = np.exp(-0.5 * (t-t0)**2 / width**2)
    return window

@function
def powerlaw_t(f, toas, log10_A=-16., gamma=5., par_0=0., par_1=0., components=2):
    Tspan = toas.max() - toas.min()
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    df = np.repeat(df, 2)
    log10_A_t = g_chebyshev(par_0, toas, Tspan)
    gamma_t = g_chebyshev(par_1, toas, Tspan)
    plaw = ((10**log10_A)** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * df)**0.5
    f = f[::components]
    weight_t = (10**log10_A_t[:, None]) * (f[None, :] / const.fyr) ** (-0.5*gamma_t[:, None])
    F = np.zeros((len(toas), 2*len(f)))
    F[:, ::2] = weight_t * np.sin(2 * np.pi * toas[:, None] * f[None, :])
    F[:, 1::2] = weight_t * np.cos(2 * np.pi * toas[:, None] * f[None, :])
    phi_weights = np.dot(F.T, F) * np.outer(plaw, plaw)
    return phi_weights

@function
def identity_sectrum(f, components=2):
    null_spec = np.ones(len(f[::components]))
    return (
        null_spec
    )

def time_correlated_block_t(psd='powerlaw_t', prior='log-uniform', Tspan=None, name='', sample_knot_t=False,
                    components=30, gamma_val=None, coefficients=False, idx=0, logf=False, spline=False,
                    combine=True, order=1, par_0_val=None, par_1_val=None, fmin=None, fmax=None,
                    logmin=None, logmax=None):

    # red noise parameters that are common
    # parameters shared by PSD functions
    logmin = -18
    logmax = -10
    if logmin is not None and logmax is not None:
        if prior == 'uniform':
            log10_A = parameter.LinearExp(logmin, logmax)
        elif prior == 'log-uniform':
            log10_A = parameter.Uniform(logmin, logmax)
    else:
        if prior == 'uniform':
            log10_A = parameter.LinearExp(-20, -11)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_A = parameter.Uniform(-20, -11)
            else:
                log10_A = parameter.Uniform(-20, -11)
        else:
            log10_A = parameter.Uniform(-20, -11)

    if gamma_val is not None:
        gamma = parameter.Constant(gamma_val)
    else:
        gamma = parameter.Uniform(0, 7)

    if psd == 'powerlaw_spline_knot_t':

        # par_0 : knot time (x coordinate)
        # par_1 : knot amp (y coordinate)

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        if par_0_val is None:
            if order > 1:
                if sample_knot_t:
                    ts = np.linspace(0, 1, order+2)
                    dt = ts[1]-ts[0]
                    par_0 = parameter.Uniform(0., dt, size=2*(order-1))
                else:
                    par_0 = None
            else:
                par_0 = None
            # par_0 = parameter.Normal(0, 1, size=order)
        else:
            par_0 = parameter.Constant(par_0_val)
        if par_1_val is None:
            # par_1 = parameter.Uniform(-5., 5., size=2*order)
            par_1 = parameter.Normal(0, 1, size=2*order)
        else:
            par_1 = parameter.Constant(par_1_val)

    if psd == 'powerlaw_spline_t':

        # par_0 : knot time (x coordinate)
        # par_1 : knot amp (y coordinate)

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        if par_0_val is None:
            par_0 = parameter.Normal(0, 1, size=order)
        else:
            par_0 = parameter.Constant(par_0_val)
        if par_1_val is None:
            # par_1 = parameter.Uniform(-5., 5., size=2*order)
            par_1 = parameter.Normal(0, 1, size=order)
        else:
            par_1 = parameter.Constant(par_1_val)

    if psd == 'powerlaw_cos_t':

        # par_0 : knot time (x coordinate)
        # par_1 : knot amp (y coordinate)

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)

    if psd == 'powerlaw_t':

        # par_0 : amplitude chebyshev polynomial coefficients
        # par_1 : gamma chebyshev polynomial coefficients

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        if par_0_val is None:
            # par_0 = parameter.Uniform(-5., 5., size=order)
            par_0 = parameter.Normal(0, 1, size=order)
        else:
            par_0 = parameter.Constant(par_0_val)
        if par_1_val is None:
            # par_1 = parameter.Uniform(-5., 5., size=order)
            par_1 = parameter.Normal(0, 1, size=order)
        else:
            par_1 = parameter.Constant(par_1_val)

    elif psd == 'spectrum_t':

        # par_0 : chebyshev polynomial coefficients for each frequency component (par_0[n*order:(n+1)*order] for component n)

        log10_rho = parameter.Uniform(-10, -4, size=components)
        pl = gpp.free_spectrum(log10_rho=log10_rho)
        # par_0 = [parameter.Uniform(-5., 5., size=order)('beta_'+str(n)) for n in range(components)]
        par_0 = parameter.Normal(0, 1, size=order*components)
        par_1 = None

    elif psd == 'spectrum_window_t':

        # par_0 : time position of window
        # par_1 : width of window

        log10_rho = parameter.Uniform(-10, -4, size=components)
        pl = gpp.free_spectrum(log10_rho=log10_rho)
        par_0 = parameter.Uniform(0., 1., size=components)
        par_1 = parameter.Uniform(0., 1., size=components)

        
    # make red noise
    rn = NonStatBasisGP(pl, psd=psd, par_0=par_0, par_1=par_1,
                                    components=components, order=order,
                                    Tspan=Tspan, idx=idx,
                                    combine=combine, logf=logf,
                                    coefficients=coefficients,
                                    name=name
                                    )

    return rn


def common_time_correlated_block_t(psd='powerlaw_t', prior='log-uniform', orf=None, Tspan=None, name='gw', sample_knot_t=False,
                    components=30, gamma_val=None, coefficients=False, idx=0, logf=False, spline=False,
                    combine=True, order=1, par_0_val=None, par_1_val=None,
                    logmin=None, logmax=None):

    orfs = {'crn': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_hd': model_orfs.param_hd_orf(a=parameter.Uniform(-1.5, 3.0)('gw_orf_param0'),
                                                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param1'),
                                                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param2')),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(-0.9, 0.9, size=7)('gw_orf_spline')),
            'bin_orf': model_orfs.bin_orf(params=parameter.Uniform(-1.0, 1.0, size=7)('gw_orf_bin'))}

    # red noise parameters that are common
    # parameters shared by PSD functions
    amp_name = '{}_log10_A'.format(name)
    if logmin is not None and logmax is not None:
        if prior == 'uniform':
            log10_A = parameter.LinearExp(logmin, logmax)(amp_name)
        elif prior == 'log-uniform':
            log10_A = parameter.Uniform(logmin, logmax)(amp_name)
    else:
        if prior == 'uniform':
            log10_A = parameter.LinearExp(-18, -10)(amp_name)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_A = parameter.Uniform(-18, -10)(amp_name)
            else:
                log10_A = parameter.Uniform(-18, -10)(amp_name)
        else:
            log10_A = parameter.Uniform(-18, -10)(amp_name)

    gam_name = '{}_gamma'.format(name)
    if gamma_val is not None:
        gamma = parameter.Constant(gamma_val)(gam_name)
    else:
        gamma = parameter.Uniform(0, 7)(gam_name)

    if psd == 'powerlaw_spline_t':

        # par_0 : knot time (x coordinate)
        # par_1 : knot amp (y coordinate)
        par_0_name = '{}_a_amp'.format(name)
        par_1_name = '{}_b_gam'.format(name)

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        if par_0_val is None:
            par_0 = parameter.Normal(0, 1, size=order)(par_0_name)
        else:
            par_0 = parameter.Constant(par_0_val)
        if par_1_val is None:
            # par_1 = parameter.Uniform(-5., 5., size=2*order)
            par_1 = parameter.Normal(0, 1, size=order)(par_1_name)
        else:
            par_1 = parameter.Constant(par_1_val)

    if psd == 'powerlaw_cos_t':

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        par_0 = None
        par_1 = None

    if psd == 'powerlaw_corr_spline_t':

        # par_0 : knot time (x coordinate)
        # par_1 : knot amp (y coordinate)

        # pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        if par_0_val is None:
            par_0 = parameter.Normal(0, 1, size=order)
        else:
            par_0 = parameter.Constant(par_0_val)
        if par_1_val is None:
            # par_1 = parameter.Uniform(-5., 5., size=2*order)
            par_1 = parameter.Normal(0, 1, size=order)
        else:
            par_1 = parameter.Constant(par_1_val)
        pl = powerlaw_t(log10_A=log10_A, gamma=gamma, par_0=par_0, par_1=par_1)

    if psd == 'powerlaw_t':

        # par_0 : amplitude chebyshev polynomial coefficients
        # par_1 : gamma chebyshev polynomial coefficients

        par_0_name = '{}_a_amp'.format(name)
        par_1_name = '{}_b_gam'.format(name)

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        if par_0_val is None:
            # par_0 = parameter.Uniform(-5., 5., size=order)
            par_0 = parameter.Normal(0, 1, size=order)(par_0_name)
        else:
            par_0 = parameter.Constant(par_0_val)
        if par_1_val is None:
            # par_1 = parameter.Uniform(-5., 5., size=order)
            par_1 = parameter.Normal(0, 1, size=order)(par_1_name)
        else:
            par_1 = parameter.Constant(par_1_val)

    elif psd == 'spectrum_t':

        # par_0 : chebyshev polynomial coefficients for each frequency component (par_0[n*order:(n+1)*order] for component n)
        
        par_0_name = '{}_a_amp'.format(name)
        rho_name = '{}_log10_rho'.format(name)
        log10_rho = parameter.Uniform(-10, -4, size=components)(rho_name)
        pl = gpp.free_spectrum(log10_rho=log10_rho)
        # par_0 = [parameter.Uniform(-5., 5., size=order)('beta_'+str(n)) for n in range(components)]
        par_0 = parameter.Normal(0., 1., size=order*components)(par_0_name)
        par_1 = None

    elif psd == 'spectrum_spline_t':

        # par_0 : spline coefficients for each frequency component (par_0[n*order:(n+1)*order] for component n)
        
        par_0_name = '{}_a_amp'.format(name)
        rho_name = '{}_log10_rho'.format(name)
        log10_rho = parameter.Uniform(-10, -4, size=components)(rho_name)
        pl = gpp.free_spectrum(log10_rho=log10_rho)
        # par_0 = [parameter.Uniform(-5., 5., size=order)('beta_'+str(n)) for n in range(components)]
        par_0 = parameter.Normal(0., 1., size=order*components)(par_0_name)
        par_1 = None
    
    # make red noise
    if orf in [None, 'crn']:
        crn = NonStatBasisGP(pl, psd=psd, par_0=par_0, par_1=par_1,
                            components=components, order=order,
                            Tspan=Tspan, idx=idx,
                            combine=combine, logf=logf,
                            coefficients=coefficients,
                            name=name
                            )
    else:
        crn = NonStatBasisCommonGP(pl, orf=orfs[orf], par_0=par_0, par_1=par_1,
                                components=components, order=order,
                                Tspan=Tspan, idx=idx,
                                combine=combine, logf=logf,
                                coefficients=coefficients,
                                name=name
                                )

    return crn

@function
def createfourierdesignmatrix_red_t(
    toas, freqs, psd='powerlaw_t', par_0=None, par_1=None, nmodes=30, order=1, idx=0., fref=1400, Tspan=None, logf=False, fmin=None, fmax=None, pshift=False, modes=None, pseed=None):
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

    # Initialize null parameter values to avoid crash during model initialization when function is called for the first time
    # need to find more elegant fix for the future
    if par_0 is None:
        if psd == 'powerlaw_t':
            par_0 = [0.]*order
        elif psd == 'powerlaw_spline_t' or psd == 'powerlaw_corr_spline_t':
            # par_0 = [None]*(2*(order-1))
            par_0 = np.zeros(order)
        elif 'spectrum_t':
            par_0 = [0.]*(nmodes*order)
        elif 'spectrum_spline_t':
            par_0 = [None]*(nmodes*(order-1))
    if par_1 is None:
        if psd == 'powerlaw_t':
            par_1 = [0.]*order
        elif psd == 'powerlaw_spline_t' or psd == 'powerlaw_corr_spline_t':
            # par_1 = np.zeros(2*order)
            par_1 = np.zeros(order)
        else:
            par_1 = [0.]*nmodes

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

    # f = np.arange(1, nmodes+1)/Tspan

    if psd == 'powerlaw_spline_knot_t':

        # par_0 : knot time postitions (par_0[::2] for amplitude, par_0[1::2] for gamma)
        # par_1 : knot amplitudes (par_1[::2] for amplitude, par_1[1::2] for gamma)

        log10_A_t = g_spline(par_0[::2], par_1[::2], order, toas, Tspan)
        gamma_t = g_spline(par_0[1::2], par_1[1::2], order, toas, Tspan)

        weight_t = (10**log10_A_t[:, None]) * (f[None, :] / const.fyr) ** (-0.5*gamma_t[:, None])
    
    if psd == 'powerlaw_spline_t':

        # par_0 : knot amplitude log10_A
        # par_1 : knot amplitudes gamma

        log10_A_t = g_spline([None]*order, par_0, order, toas, Tspan)
        gamma_t = g_spline([None]*order, par_1, order, toas, Tspan)

        weight_t = (10**log10_A_t[:, None]) * (f[None, :] / const.fyr) ** (-0.5*gamma_t[:, None])

    if psd == 'powerlaw_cos_t':

        weight_t = (1 - np.cos(2*np.pi*f[None, :] * toas[:, None]))

    elif psd == 'spectrum_spline_t':

        # par_0 : chebyshev polynomial coefficients for each frequency component (par_0[n*order:(n+1)*order] for component n)
        
        log10_weight_t = np.array([g_spline([None]*order, par_0[n*order:(n+1)*order], order, toas, Tspan) for n in range(nmodes)]).T
        # log10_weight_t = np.array([g_chebyshev(par_0[n*order:(n+1)*order], toas, Tspan) for n in range(nmodes)]).T
        weight_t = 10**log10_weight_t

    elif psd == 'powerlaw_t':

        # par_0 : amplitude chebyshev polynomial coefficients
        # par_1 : gamma chebyshev polynomial coefficients

        log10_A_t = g_chebyshev(par_0, toas, Tspan)
        gamma_t = g_chebyshev(par_1, toas, Tspan)
        weight_t = (10**log10_A_t[:, None]) * (f[None, :] / const.fyr) ** (-0.5*gamma_t[:, None])

    elif psd == 'spectrum_t':

        # par_0 : chebyshev polynomial coefficients for each frequency component (par_0[n*order:(n+1)*order] for component n)

        log10_weight_t = np.array([g_chebyshev(par_0[n*order:(n+1)*order], toas, Tspan) for n in range(nmodes)]).T
        weight_t = 10**log10_weight_t

    elif psd == 'spectrum_window_t':

        # par_0 : time window position
        # par_1 : time window width

        weight_t = np.array([exp_window(beta_a, beta_g, toas, Tspan) for beta_a, beta_g in zip(par_0, par_1)]).T

    # The sine/cosine modes
    # The sine/cosine modes
    F[:, ::2] = weight_t * np.sin(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = weight_t * np.cos(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])

    if idx > 0.:
    # compute the DM-variation vectors
        Dm = (fref / freqs) ** idx
        F *= Dm[:, None]

    return F, Ffreqs

def NonStatBasisGP(
    spectrum,
    psd='powerlaw_t',
    par_0=None,
    par_1=None,
    Tspan=None,
    coefficients=False,
    combine=True,
    components=20,
    order=1,
    idx=0,
    logf=False,
    selection=Selection(selections.no_selection),
    name="red_noise",
):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = createfourierdesignmatrix_red_t(psd=psd, par_0=par_0, par_1=par_1, nmodes=components, order=order, Tspan=Tspan, idx=idx, logf=logf)
    BaseClass = gp_signals.BasisGP(spectrum, basis, coefficients=coefficients, combine=combine, selection=selection, name=name)

    class FourierBasisGP(BaseClass):
        signal_type = "basis"
        signal_name = 'non stat '+name
        signal_id = name

        # def __init__(self, psr):
        #     super(FourierBasisGP, self).__init__(psr)

        # def _construct_basis(self, params={}):
        #     self._basis, self._labels = self._bases(params=params, Tspan=Tspan)

    return FourierBasisGP

def NonStatBasisCommonGP(
    spectrum,
    orf,
    psd='powerlaw_t',
    par_0=None,
    par_1=None,
    coefficients=False,
    combine=True,
    logf=False,
    components=20,
    Tspan=None,
    order=1,
    idx=0,
    modes=None,
    name="common_fourier",
    pshift=False,
    pseed=None
):

    if coefficients and Tspan is None:
        raise ValueError(
            "With coefficients=True, FourierBasisCommonGP " + "requires that you specify Tspan explicitly."
        )

    basis = createfourierdesignmatrix_red_t(psd=psd, par_0=par_0, par_1=par_1, nmodes=components, order=order, Tspan=Tspan, idx=idx, logf=logf)
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