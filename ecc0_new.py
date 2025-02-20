import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline as spline 
from scipy.interpolate import InterpolatedUnivariateSpline as univ_spline
import scipy.constants as const
from enterprise.signals import (signal_base, parameter, deterministic_signals)
import scipy.special as ss
from scipy.integrate import odeint
#from numba import njit
#from enterprise_extensions.deterministic import CWSignal

G = const.G
c = const.c
msun = 2*1e30*G/c**3
kpc = 1e3*const.parsec

def t_p(t, Lp, cos_mu):
    
    tau_p = Lp*kpc/c*(1-cos_mu)
    tp = t - tau_p
    print(Lp, tp.min(), kpc/c)
    return tp
    

def xdot(x, M, nu, e, xi):
    #dxdt = 2/3*(nu*(96 + 292*e**2 + 37*e**4)*x**(5))/(5*M*(1 - e**2)**(7/2)) + (2*nu*(-576 + 111*e**6*(-3 + nu) + 8*e**2*(-690 + 157*nu) + 6*e**4*(-841 + 268*nu))*x**6)/(15*(1 - e**2)**(9/2)*M)
    dxdt = 2/3*(nu*(96 + 292*e**2 + 37*e**4)*x**(5))/(5*M*(1 - e**2)**(7/2)) -(((nu*(16*(743 + 924*nu) + e**6*(6931 + 2072*nu) + 14*e**4*(7079 + 3690*nu) + 8*e**2*(15411 + 11158*nu)))*x**6)/(420*((1 - e**2)**(9/2)*M)))

    return dxdt

def edot(x, M, nu, e, xi):
    #dedt = -(e*nu*(304 + 121*e**2)*x**(4))/(15*M*(1 - e**2)**(5/2)) - ((e*nu*(-2432 + 363*e**4*(-3 + nu) +2*e**2*(-3127 + 881*nu)))*x**5)/(15*((1- e**2)**(7/2)*M))
    dedt = -(e*nu*(304 + 121*e**2)*x**(4))/(15*M*(1 - e**2)**(5/2)) +((e*nu*(e**4*(94887 + 19768*nu) + 12*e**2*(38698 + 21427*nu) + 8*(20547 + 24556*nu))*x**5)/( 2520*(1 - e**2)**(7/2)*M))
    
    return dedt


def xdot_0(x, M, nu, e, xi):
    dxdt = 2/3*(nu*(96 + 292*e**2 + 37*e**4)*x**(5))/(5*M*(1 - e**2)**(7/2)) 

    return dxdt

def edot_0(x, M, nu, e, xi):
    dedt = -(e*nu*(304 + 121*e**2)*x**(4))/(15*M*(1 - e**2)**(5/2))
    return dedt


def get_phidot(x, M, nu, e, xi):
    p = (1-e**2)/x + 1/3 *(nu +e**2*(6-nu))
    dphidt = ((1+e*np.cos(xi))**2/(M*p**(3/2)))*(1-(2*e*np.cos(xi))/p +(nu*(1-e**2))/(2*p))
    #dphidt = ((1+e*np.cos(xi))**2/(M*p**(3/2)))
    return dphidt

def get_xidot(x, M, nu, e, xi):
    p = (1-e**2)/x + 1/3 *(nu +e**2*(6-nu))
    dxidt = ((1+e*np.cos(xi))**2/(M*p**(3/2)))*(1-(3*(1+e*np.cos(xi))/p)+(nu*(1-e**2))/(2*p))
    #dxidt = ((1+e*np.cos(xi))**2/(M*p**(3/2)))
    return dxidt

def get_gammadot(x, M, nu, e, xi):
    return get_phidot(x, M, nu, e, xi) - get_xidot(x, M, nu, e, xi)
    #return ((6+5*e**2)*np.pi*x**(5/2)/(M*(1 - e**2)) - (5*((6 + 5*e**2)*np.pi*(-6*e**2 - nu + e**2*nu))*x**(7/2))/(6*((-1 + e)*(1 + e)*M*((1 - e**2))**(5/2))))/(2*np.pi)
    
def get_gammadot_0(x, M, nu, e, xi):
    return get_phidot(x, M, nu, e, xi) - get_xidot(x, M, nu, e, xi)
    #return 3*x**(5/2)/(M*(1 - e**2)) + 3*(-5 +2*e**2*(-3+nu))*x**(7/2)/(M*(1-e**2))

def ode_system(y, t, M, nu):
    
    e, x, xi, gamma = y
    
    dedt = edot(x, M, nu, e, xi)
    dxdt = xdot(x, M, nu, e, xi)
    dxidt = get_xidot(x, M, nu, e, xi)
    dgammadt = get_gammadot(x, M, nu, e, xi)

    return np.array([dedt, dxdt, dxidt, dgammadt])


def ode_system_0(y, t, M, nu):
    
    e, x, xi, gamma = y
    
    dedt = edot(x, M, nu, e, xi)
    dxdt = xdot(x, M, nu, e, xi)
    dxidt = get_xidot(x, M, nu, e, xi)
    dgammadt = get_gammadot_0(x, M, nu, e, xi)

    return np.array([dedt, dxdt, dxidt, dgammadt])

def full_ode_system(y, t, M, nu, D, cos_inc):
    
    e, x, xi, gamma, rplus, rcross = y
    
    dedt = edot(x, M, nu, e, xi)
    dxdt = xdot(x, M, nu, e, xi)
    dxidt = get_xidot(x, M, nu, e, xi)
    dgammadt = get_gammadot(x, M, nu, e, xi)
    hplus_d = h_plus(x, M, nu, D, cos_inc, e, xi, gamma)
    hcross_d = h_cross(x, M, nu, D, cos_inc, e, xi, gamma)
    
    return np.array([dedt, dxdt, dxidt, dgammadt, hplus_d, hcross_d])

#@njit
def h_plus(x, M, nu, D, cos_inc, e, xi, gamma):
    integrand = (x*nu*M/(D*(1-e**2)))*((1 + cos_inc**2)*((e**2 + 5/2*e*np.cos(xi) +2*np.cos(2*xi) + e/2*np.cos(3*xi))*np.cos(2*gamma) - (5/2*e*np.sin(xi) + 2*np.sin(2*xi)+e/2*np.sin(3*xi))*np.sin(2*gamma)) + (1 - cos_inc**2)*e*(e+np.cos(xi)))

    return integrand

#@njit
def h_cross(x, M, nu, D, cos_inc, e, xi, gamma):
    integrand = (x*nu*M/(D*(1-e**2)))*((cos_inc*2)*((e**2 + 5/2*e*np.cos(xi) +2*np.cos(2*xi) + e/2*np.cos(3*xi))*np.sin(2*gamma) + (5/2*e*np.sin(xi) + 2*np.sin(2*xi)+e/2*np.sin(3*xi))*np.cos(2*gamma)))

    return integrand 
'''
def xi_t(M, x, xi0, omega_r, nu, e, toas):

    xi = np.linspace(xi0, xi0 + omega_r*(toas.max() -toas.min()), int(1e4))
    
    p = (1-e**2)/x + (1/3)*(nu +e**2*(6-nu))
    a = 2*(6 + 2*p + nu - e**2*(6 + nu))*(np.arctan(np.sqrt((1-e)/(1 + e))*np.tan(xi/2))%(np.pi) -(xi/2)%np.pi + xi/2)/(1 - e**2)**(3/2)
    b = -(72*(np.arctan((np.sqrt(-(-6 + 2*p + nu + e*(6 - e*nu))/(6 - 2*p - nu + e*(6 + e*nu)))*np.tan(xi/2)))%(np.pi) -(xi/2)%np.pi + xi/2)/np.sqrt(((-6 + 2*p + nu + e*(6 - e*nu))*(-6 + 2*p + nu - e*(6 + e*nu)))))
    c = -(e*(2*p + nu - e**2*nu)*np.sin(xi))/((1 - e**2)*(1 + e*np.cos(xi)))

    t =(((2*M*p**(5/2))/(2*p + nu - nu*e**2)**2)*(a+b+c))
    
    spline_xi = CubicSpline(t+toas.min(), xi + xi0)
    
    return spline_xi(toas)
'''
'''
def xi_t(M, x, xi0, omega_r, nu, e, toas):
    #print(M)
    xi =xi0 + np.linspace(0, omega_r*(toas.max() - toas.min()), int(8e3))
    
    p = (1-e**2)/x + (1/3)*(nu +e**2*(6-nu))
    a = 2*(6 + 2*p + nu - e**2*(6 + nu))*(np.arctan(np.sqrt((1-e)/(1 + e))*np.tan(xi/2))%(np.pi) -(xi/2)%np.pi + xi/2)/(1 - e**2)**(3/2)
    b = -(72*(np.arctan((np.sqrt(-(-6 + 2*p + nu + e*(6 - e*nu))/(6 - 2*p - nu + e*(6 + e*nu)))*np.tan(xi/2)))%(np.pi) -(xi/2)%np.pi + xi/2)/np.sqrt(((-6 + 2*p + nu + e*(6 - e*nu))*(-6 + 2*p + nu - e*(6 + e*nu)))))
    c = -(e*(2*p + nu - e**2*nu)*np.sin(xi))/((1 - e**2)*(1 + e*np.cos(xi)))

    t =(((2*M*p**(5/2))/(2*p + nu - nu*e**2)**2)*(a+b+c)) 
    
    #spline_xi = spline(t-t[0], xi) 
    print((t-t[0])[0])
    print(xi[0])
    
    #return spline_xi(toas - toas.min()), t, xi
    return t, xi
'''
#@njit
def xi_t(M, x, xi0, omega_r, nu, e, toas):
    #print(M)
    xi =xi0 + np.linspace(0, omega_r*(toas.max() - toas.min()), int(1e3))
    
    p = (1-e**2)/x + (1/3)*(nu +e**2*(6-nu))
    a = 2*(6 + 2*p + nu - e**2*(6 + nu))*(np.unwrap(np.arctan(np.tan(xi/2)*(np.sqrt((1-e)/(1+e)))),period = np.pi))/(1 - e**2)**(3/2)
    b = -(72*(np.unwrap(np.arctan((np.sqrt(-(-6 + 2*p + nu + e*(6 - e*nu))/(6 - 2*p - nu + e*(6 + e*nu)))*np.tan(xi/2))),period = np.pi))/np.sqrt(((-6 + 2*p + nu + e*(6 - e*nu))*(-6 + 2*p + nu - e*(6 + e*nu)))))
    c = -(e*(2*p + nu - e**2*nu)*np.sin(xi))/((1 - e**2)*(1 + e*np.cos(xi)))

    t =(((2*M*p**(5/2))/(2*p + nu - nu*e**2)**2)*(a+b+c)) 
    
    spline_xi = spline(t-t[0], xi) 
    #spline_xi = univ_spline(t-t[0], xi)
    #print((t-t[0])[0])
    #print(xi[0])
    
    return spline_xi(toas-toas[0])
    #return t, xi


def xi_t_01(M, x, xi0, omega_r, nu, e, toas, order):
    
    xi = xi0+ np.linspace(0, omega_r*(toas.max() - toas.min()), int(1e4))
    t_0 = ((np.sqrt(1-e**2)*M)/x**(3/2))*((2*np.unwrap(np.arctan(np.tan(xi/2)*(np.sqrt((1-e)/(1+e)))),period = np.pi))/(np.sqrt(1-e**2)) - (e*np.sin(xi))/(1+e*np.cos(xi)))
    
    '''
    t_1 = ((np.sqrt(1 - e**2)*M*(-2*(-36*e**2*(-2 + e**2) - 60*(-1 + e**2)**2*nu + 5*(-1 + e**2)**2*nu**2)*((np.arctan(np.sqrt((1-e)/(1 + e))*np.tan(xi/2))%(np.pi) -(xi/2)%np.pi + xi/2))*(1 + e*np.cos(xi))+e*np.sqrt(1 - e**2)*(36*e**4 + 5*(-1 + e**2)**2*nu**2)*np.sin(xi))*np.sqrt(x)))/(24*((1 - e**2)**(5/2)*(1 + e*np.cos(xi))))
    '''
    t_1 = (3*M/np.sqrt(x))*((2*np.unwrap(np.arctan(np.tan(xi/2)*(np.sqrt((1-e)/(1+e)))),period = np.pi))/((1-e**2)) - (e**3*np.sin(xi))/(np.sqrt(1-e**2)*(1+e*np.cos(xi))))
    
    c0 = -36*e**2*(-2 + e**2) - 60*(-1 + e**2)*2*nu + 5*(-1 + e**2)**2*nu**2
    c1 = 216*(-1 + e**2)*xi*(1 + e*np.cos(xi)) +  e*(36*e**4 + 5*(-1 + e**2)**2*nu**2)*np.sin(xi)
    
    t_2 = (M*(1-e**2)**(-2)/(24*(1+e*np.cos(xi))))*(np.sqrt(1-e**2)*c1+2*c0*np.unwrap(np.arctan(np.tan(xi/2)*(np.sqrt((1-e)/(1+e)))),period = np.pi)*(1+np.cos(xi)))
    if order == 0:
        t=t_0
    if order == 1:
        t = t_0 + t_1
    else:
        t = t_0 + t_1 + t_2
        
    spline_xi = spline(t-t.min(), xi) 
    return spline_xi(toas - toas.min())


def phi_xi(M, x, phi0, nu, e, xi):
    p = (1-e**2)/x + (1/3)*(nu +e**2*(6-nu))
    a = np.sqrt(-6 + 2*p + nu + e*(6 - e*nu))
    b = np.sqrt(-6 + 2*p + nu - e*(6 + e*nu))
    xi_n = xi - xi.min()
    return phi0 + 2/3*(xi_n + ((12 + 2*p + nu - e**2*nu)/(a*b)*np.unwrap(np.arctan(a/b*np.tan(xi_n/2)),period = np.pi))) 



def res_analit_plus(e, gamma, t, Mc, D, xi, iota, omega_phi, omega_r): 
    #print('omega_phi in res = ', omega_phi)
    #coeff = (Mc**(5/3)*np.sqrt(1-e**2))/(D*omega_phi**(1/3))
    coeff = (Mc**(5/3)*np.sqrt(1-e**2))*omega_phi**(2/3)/(D*omega_r)
    a = (e+2*np.cos(xi))*np.sin(xi)/(1+e*np.cos(xi))
    b = (np.cos(2*xi) + e* np.cos(xi))/(1+e*np.cos(xi))
    c = (e*np.sin(xi)/(1+e*np.cos(xi)))
    return coeff*((1+np.cos(iota)**2)*(a*np.cos(2*gamma)+b*np.sin(2*gamma)) + np.sin(iota)**2*c) 
    #return ((1+np.cos(iota)**2)*(a*np.cos(2*gamma)+b*np.sin(2*gamma))) 


def res_analit_cross(e, gamma, t, Mc, D, xi, iota, omega_phi, omega_r):
    #coeff = (Mc**(5/3)*np.sqrt(1-e**2))/(D*omega_phi**(1/3))
    coeff = (Mc**(5/3)*np.sqrt(1-e**2))*omega_phi**(2/3)/(D*omega_r)
    a = (e+2*np.cos(xi))*np.sin(xi)/(1+e*np.cos(xi))
    b = (np.cos(2*xi) + e* np.cos(xi))/(1+e*np.cos(xi))
    return coeff*(2*np.cos(iota))*(a*np.sin(2*gamma)-b*np.cos(2*gamma))


def psi_r(e, xi):
    return 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(xi/2)) - e * np.sqrt(1-e**2)*np.sin(xi)/(1+e*np.cos(xi))
    

def res_plus(e, gamma, t, M, D, nu, xi, iota, omega_phi, omega_r, k_arr):
    coeff = (M)**(5/3)*(1-e**2)**(1/2)*nu/(D*omega_phi**(1/3))
    k_piece1 = np.array([(2/k *ss.jn(k, k*e) - 4/(e**2*k)*ss.jn(k, k*e) +2/e*ss.jn(k-1,k*e) - 2/e*ss.jn(k+1,k*e)-2*e*ss.jn(k-1,k*e)+ 2*e*ss.jn(k+1,k*e))/np.sqrt(1-e**2)*np.sin(k*psi_r(e,xi)) for k in k_arr])
    k_piece2 = np.array([(4*(1-e**2)/e**2 *ss.jn(k, k*e) - 2/(e*k)*(ss.jn(k-1,k*e) - ss.jn(k+1,k*e)))*np.cos(k*psi_r(e,xi)) for k in k_arr])
    k_piece3 = np.array([e/(np.sqrt(1-e**2)*k)*(ss.jn(k-1, k*e) +ss.jn(k+1,k*e))*np.sin(k*psi_r(e,xi)) for k in k_arr])
    return coeff*((1+np.cos(iota)**2)*(np.cos(2*gamma)*(np.sum(k_piece1,axis =0))+ np.sin(2*gamma)*(np.sum(k_piece2,axis =0))) + np.sin(iota)**2*np.sum(k_piece3,axis =0))

def res_cross(e, gamma, t, M, D, nu, xi, iota, omega_phi, omega_r, k_arr):
    coeff = (M)**(5/3)*(1-e**2)**(1/2)*nu/(D*omega_phi**(1/3))
    k_piece1 = np.array([(2/k *ss.jn(k, k*e) - 4/(e**2*k)*ss.jn(k, k*e) +2/e*ss.jn(k-1,k*e) - 2/e*ss.jn(k+1,k*e)-2*e*ss.jn(k-1,k*e)+ 2*e*ss.jn(k+1,k*e))/np.sqrt(1-e**2)*np.sin(k*psi_r(e,xi)) for k in k_arr])
    k_piece2 = np.array([(4*(1-e**2)/e**2 *ss.jn(k, k*e) - 2/(e*k)*(ss.jn(k-1,k*e) - ss.jn(k+1,k*e)))*np.cos(k*psi_r(e,xi)) for k in k_arr])
    return coeff*((2*np.cos(iota))*(np.sin(2*gamma)*(np.sum(k_piece1,axis =0))-  np.cos(2*gamma)*(np.sum(k_piece2,axis =0))))



def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([-np.sin(gwphi), np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta) * np.cos(gwphi), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)])

    fplus = 0.5 * (np.dot(m, pos) ** 2 - np.dot(n, pos) ** 2) / (1 + np.dot(omhat, pos))
    fcross = (np.dot(m, pos) * np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu
'''
def create_gw_antenna_pattern(pos, gwtheta, gwphi, psi):
    """
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([np.sin(gwphi)*np.cos(psi) - np.sin(psi)*np.cos(gwphi)*np.cos(gwtheta), -(np.cos(gwphi)*np.cos(psi)+ np.sin(psi)*np.sin(gwphi)*np.cos(gwtheta)), np.sin(psi)*np.sin(gwtheta)])
    n = np.array([-np.sin(gwphi)*np.sin(psi) - np.cos(psi)*np.cos(gwphi)*np.cos(gwtheta), (np.cos(gwphi)*np.sin(psi)- np.cos(psi)*np.sin(gwphi)*np.cos(gwtheta)), np.cos(psi)*np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)])


    fplus = 0.5 * (np.dot(m, pos) ** 2 - np.dot(n, pos) ** 2) / (1 + np.dot(omhat, pos))
    fcross = (np.dot(m, pos) * np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu

'''

@signal_base.function
def analytic_eccentric_residuals(toas, pos, pdist, cos_gwtheta, gwphi,
                                log10_Mc, log10_dist, log10_F, cos_inc,
                                psi, gamma0, xi0, e0, nu, analytic, psrTerm=False, tref=0):
    
    '''
    Residuals in TOAs produced by GW emitted by eccentric SMBHB. 
    Waveform model from Hinderer, Babak 2017.
    
    :param toa: pulsar observation times
    :param theta_p: polar coordinate of pulsar
    :param phi_p: azimuthal coordinate of pulsar
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param log10_M: Base-10 lof of total mass of SMBMB [solar masses]
    :param log10_dist: Base-10 uminosity distance to SMBMB [Mpc]
    :param log10_F: base-10 orbital frequency of SMBHB [Hz]
    :param inc: Inclination of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param phi0: Initial orbital phase [radians]
    :param xi0: Initial true anomaly [readians]
    :param e0: Initial eccentricity of SMBHB
    :param nu: Symmetric mass ratio of SMBHB
    :param psrTerm: Option to include pulsar term [boolean]
    :param tref: Time at which initial parameters are referenced [s]

    :returns: Vector of induced residuals

    '''

    # convert from sampling
    F = 10.0**log10_F
    Mc = 10.0**log10_Mc*msun
    M = Mc*nu**(-3/5)
    D = (10.0**log10_dist)/c*(1e6*const.parsec)
    
    iota = np.arccos(cos_inc)
    gwtheta = np.arccos(cos_gwtheta)

    t = toas - tref
    #print('t min = ', t.min())
    #theta = np.pi/2 - dec_p
    #phi = ra_p
    ##print(t)
    
    omega_phi0 = 2*np.pi*F
    x0 = (M*omega_phi0)**(2/3)
    p0 = (1-e0**2)/x0 + (1/3)*(nu +e0**2*(6-nu))
    omega_r0 = (1-e0**2)**(3/2)/(M*p0**(3/2))*(1+((1-e0**2)*(-6+nu))/(2*p0)) #omega_phi0 - 3*(1-e0**2)**(3/2)/(M*p0**(5/2)) 
    #print('e0 = ' + str(e0))
    #print(x0)
    #print(M)
    e = e0 + edot(x0, M, nu, e0, xi0)*(t-t[0])
    x = x0 + xdot(x0, M, nu, e, xi0)*(t-t[0])
    
    phi0 = gamma0+xi0
    
    #gamma_dot = phi_dot - xi_dot

    p = (1-e**2)/x + (1/3)*(nu +e**2*(6-nu))
    omega_r = (1-e**2)**(3/2)/(M*p**(3/2))*(1+((1-e**2)*(-6+nu))/(2*p))
    omega_phi = x**(3/2)/M
    xi = xi_t(M, x0, xi0, omega_r0, nu, e0, t)
    
    #phi = phi_xi(M, x, phi0, nu, e, xi)
    
    gamma = gamma0+ (omega_phi0-omega_r0)*(t-t[0])
    # pulsar position vector
    phat = pos #np.array([np.sin(theta_p)*np.cos(phi_p), np.sin(theta_p)*np.sin(phi_p),p.cos(theta_p)])
    
    #antenna pattern function
    Fplus, Fcross, cos_mu = create_gw_antenna_pattern(phat, gwtheta, gwphi)
    
    ##### earth term #####
    k_arr = np.arange(1,20)
    if analytic == True:
        splus = res_analit_plus(e, gamma, t, Mc, D, xi, iota, omega_phi, omega_r)
        scross = res_analit_cross(e, gamma, t, Mc, D, xi, iota, omega_phi, omega_r)
    else:
        splus = res_plus(e0, gamma, t, M, D, nu, xi, iota, omega_phi, omega_r, k_arr)
        scross = res_cross(e0, gamma, t, M, D, nu, xi, iota, omega_phi, omega_r, k_arr)

    delta_splus = splus
    delta_scross = scross
    
    if psrTerm == True:
        #toas at pulsar time
        tpulsar = t_p(t, pdist, cos_mu)
        
        tp_arr = np.linspace(t.min(), tpulsar.min(), int(1e4)).astype(np.float64)
        
        #numerical solution for eccentricity and frequency evolution
        y0 = np.array([e0, x0, xi0, gamma0]).astype(np.float64)
        y, infodict = odeint(ode_system, y0, tp_arr , args=(M,nu), full_output=True)

        e0_p, x0_p, xi0_p, gamma0_p = y[-1]
        print(y[-1])
        e_p = e0_p + edot(x0_p, M, nu, e0_p, xi0_p)*(t-t[0])#np.abs(tpulsar-tpulsar.min())
        x_p = x0_p + xdot(x0_p, M, nu, e0_p, xi0_p)*(t-t[0])#np.abs(tpulsar-tpulsar.min())
        p_p = (1-e_p**2)/x_p + (1/3)*(nu +e_p**2*(6-nu))

        omega_r_p = (1-e_p**2)**(3/2)/(M*p_p**(3/2))*(1+((1-e_p**2)*(-6+nu))/(2*p_p))
        omega_phi_p = x_p**(3/2)/M
        xi_p = xi_t(M, x0_p, xi0_p, omega_r_p[0], nu, e0_p, t)
        gamma_p = gamma0_p + (omega_phi_p[0] - omega_r_p[0])*(t-t[0])#np.abs(tpulsar-tpulsar.min())
        y_p = np.array([e_p, x_p, xi_p, gamma_p])
        splus_PT = res_analit_plus(e_p, gamma_p, t, Mc, D, xi_p, iota, omega_phi_p, omega_r_p)  
        scross_PT = res_analit_cross(e_p, gamma_p, t, Mc, D, xi_p, iota, omega_phi_p, omega_r_p) 
        
        delta_splus = splus - splus_PT
        delta_scross = scross - scross_PT
    
    #print(Fplus)
    #print(Fcross)
    #residuals = Fplus*(splus * np.cos(2*psi) - scross*np.sin(2*psi)) - Fcross*(splus*np.sin(2*psi)+scross*np.cos(2*psi))
    #residuals = (Fplus * np.cos(2*psi) - Fcross * np.sin(2*psi)) * delta_splus - (Fplus * np.sin(2*psi) + Fcross * np.cos(2*psi)) * delta_scross
    residuals =  (Fplus * np.cos(2*psi) + Fcross * np.sin(2*psi)) * delta_splus - (Fplus * np.sin(2*psi) - Fcross * np.cos(2*psi)) * delta_scross
    #residuals = Fplus*delta_splus + Fcross*delta_scross
    if psrTerm == True:
        return residuals, y, y_p, tpulsar
    else:
        return residuals


def cw_block_ecc(amp_prior='log-unif:orm', skyloc=None, log10_F=None,
                 ecc=None, psrTerm=False, tref=0, name='cgwe', analytic = False):
    """
    Returns deterministic, eccentric orbit continuous GW model:

    :param amp_prior:
        Prior on log10_h and log10_Mc/log10_dL. Default is "log-uniform" with
        log10_M and log10_dL searched over. Use "uniform" for upper limits,
        log10_h searched over.
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_F:
        Fixed log-10 orbital frequency of CW signal search.
        Search over orbital frequency if ``None`` given.
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
        log10_h = None
    # Total mass [Msol]
    log10_Mc = parameter.Uniform(8, 10.0)('{}_log10_Mc'.format(name))
    # luminosity distance [Mpc]
    log10_dL = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))


    # orbital frequency [Hz]
    if log10_F is None:
        log10_Forb = parameter.Uniform(-9.0, -7.5)('{}_log10_Forb'.format(name))
    else:
        log10_Forb = parameter.Constant(log10_F)('{}_log10_Forb'.format(name))
        
    # cosine of orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    
    # initial true anomaly [radians]
    xi_0 = parameter.Uniform(0.0, 2*np.pi)('{}_xi0'.format(name))
    # initial orbital phase [radians]
    #phi_0 = parameter.Uniform(0.0, 2.0*np.pi)('{}_phi0'.format(name))
    #advance of periastron
    gamma_0 = parameter.Uniform(0.0, np.pi)('{}_gamma0'.format(name))
    # Initial Earth-term eccentricity
    if ecc is None:
        e_0 = parameter.Uniform(0.01, 0.85)('{}_e0'.format(name))
    else:
        e_0 = parameter.Constant(ecc)('{}_e0'.format(name))

    # Symmetric mass ratio (not sampled, fixed for equal mass binary)
    nu = parameter.Constant(0.25)('{}_nu'.format(name))
    # nu = parameter.Uniform(0.0, 0.25)('{}_nu'.format(name))

    
    # polarization
    pol_name = '{}_pol'.format(name)
    pol = parameter.Uniform(0, np.pi)(pol_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        costheta = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costheta = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    # continuous wave signal
    wf = analytic_eccentric_residuals(cos_gwtheta=costheta, gwphi=phi,
                                     log10_Mc=log10_Mc, log10_dist=log10_dL,
                                     log10_F=log10_Forb,
                                     cos_inc=cosinc, psi=pol, gamma0=gamma_0, xi0 = xi_0,
                                     e0=e_0, nu = nu, analytic = analytic, psrTerm=False, tref=tref, pdist = None)

    cw = CWSignal(wf, ecc=True, psrTerm=psrTerm)

    return cw


def CWSignal(cw_wf, ecc=False, psrTerm=False, name='cw'):

    BaseClass = deterministic_signals.Deterministic(cw_wf, name=name)

    class CWSignal(BaseClass):

        def __init__(self, psr):
            super(CWSignal, self).__init__(psr)
            self._wf[''].add_kwarg(psrTerm=psrTerm)
            if ecc and psrTerm:
                pgam = parameter.Uniform(0, 2*np.pi)('_'.join([psr.name,
                                                               'pgam',
                                                               name]))
                self._params['pgam'] = pgam
                self._wf['']._params['pgam'] = pgam

    return CWSignal


