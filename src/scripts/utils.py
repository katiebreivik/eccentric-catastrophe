from legwork import evol, utils
import legwork as lw
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18, z_at_value

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

def dg_de(f, e):
    prefac = 76 / 243 * (1 / f) * 18/19
    numerator = (3737 * e**8 + 80225 * e**6 + 149560 * e**4 + 158832 * e**2 - 31104)
    denominator = 3 * (37 * e**4 + 292 * e**2 + 96)**2
    
    return -1 * prefac * numerator / denominator

def get_F(e):
    e1219 = e**(12/19)
    e2 = e**2
    F = e1219 / (1 - e2) * (1 + 121/304 * e2)**(870/2299)
    return F

def get_F_prime(e):
    e4 = e**4
    e2 = e**2
    e719 = e**(7/10)
    term1 = 2**(1181/2299) * 19**(870/2299)
    term2 = (121*e2 + 304)**(1429/2299)
    numerator = 37*e4 + 292*e2 + 96
    denominator = term1 * e719 * term2 * (1-e2)**2
    
    return numerator/denominator

def get_de10_deL(e10, eL):
    term1 = get_F_prime(eL) / get_F_prime(e10)
    term2 = get_F(e10) / get_F(eL)
    de10_deL =  term1 * term2
    
    return de10_deL
    
    
    
def dTmerger_df(m1, m2, f, e):
    beta = lw.utils.beta(m1, m2)
    ecc_fac = (1 - e**2)**(7/2) * (1 + 0.27 * e**10 + 0.33 * e**20 + 0.2 * e**1000)
    dt_df = -2 / 3 * (c.G * (m1 + m2) / (4 * np.pi**2))**(4/3) / beta * f**(-11/3)
    
    return ecc_fac * dt_df

def dTmerger_df_circ(m1, m2, f):
    beta = lw.utils.beta(m1, m2)
    dt_df = -2 / 3 * (c.G * (m1 + m2) / (4 * np.pi**2))**(4/3) / beta * f**(-11/3)
    
    return dt_df

def create_timesteps(t_evol=-100*u.yr, nstep_fast=1000, nstep=20):
    t_chunks = np.logspace(0, np.log10(-1*t_evol.to(u.s).value), 10)
    t_lo = t_chunks[0]
    times = np.logspace(-2, np.log10(t_lo), nstep_fast)
    timesteps = -1 * times * u.s
    for t in t_chunks[1:]:
        times = np.logspace(np.log10(t_lo), np.log10(t), nstep_fast)
        timesteps = np.append(timesteps, -1 * times * u.s)
        t_lo = t
    return timesteps

def de_dt(beta, c_0, e):
    dedt = -19 / 12 * beta / c_0**4 * (e**(-29 / 19) * (1 - e**2)**(3/2)) \
        / (1 + (121/304) * e**2)**(1181/2299)
    return dedt

def get_t_evol_LISA(dat):
    m1, m2, e_LIGO = dat

    f_LIGO=10*u.Hz
    a_LIGO = lw.utils.get_a_from_f_orb(f_orb=f_LIGO, m_1=m1, m_2=m2)

    beta = lw.utils.beta(m1, m2)
    c_0 = lw.utils.c_0(a_i = a_LIGO, ecc_i=e_LIGO)

    f_LISA=1e-5*u.Hz
    t_merge_circ = evol.get_t_merge_circ(m_1=m1, m_2=m2, f_orb_i=f_LISA).to(u.yr)
    print(t_merge_circ)

    ecc_evol = np.logspace(np.log10(e_LIGO), min(np.log10(e_LIGO)/5, np.log10(0.999)), 10000)
    dedt = de_dt(beta, c_0, e=ecc_evol)
    dedt = dedt.to(u.s**(-1))

    a = lw.utils.get_a_from_ecc(ecc_evol, c_0=c_0)
    f = lw.utils.get_f_orb_from_a(a=a, m_1=m1, m_2=m2)
    
    timesteps = cumtrapz(dedt.flatten(), ecc_evol.flatten()) * u.s
    if min(f) > f_LISA:
        print('not low enough')
        print(f.T, ecc_evol)
    elif max(f) < f_LISA:
        print('not high enough;')
        print(f.T, ecc_evol)
    
    t_interp = interp1d(f.flatten()[1:], timesteps)
    
    t_LISA = t_interp(f_LISA)

    return t_LISA

def get_t_evol_from_f(m1, m2, e_LIGO=1e-5, f_LIGO=10*u.Hz, log_f_LISA_lo=-5, log_f_LISA_hi=-1, n_f_grid=100):
    t_merge_circ = evol.get_t_merge_circ(m_1=m1, m_2=m2, f_orb_i=10**log_f_LISA_lo*u.Hz).to(u.yr)
    if e_LIGO == 0:
        timesteps = create_timesteps(t_evol = -2*t_merge_circ, nstep_fast=50 * max(1, int(e_LIGO/1e-4)))

        a_evol, f_evol = evol.evol_circ(
            m_1=m1, m_2=m2, f_orb_i=f_LIGO, timesteps=timesteps,
            output_vars=["a", "f_orb"])
    
    else:
        timesteps = create_timesteps(t_evol = -2*t_merge_circ, nstep_fast=50 * max(1, int(e_LIGO/1e-4)))
        a_evol, e_evol, f_evol = evol.evol_ecc(
            m_1=m1, m_2=m2, f_orb_i=f_LIGO, ecc_i=e_LIGO, timesteps=timesteps,
            output_vars=["a", "ecc", "f_orb"], avoid_merger=False)
        
    t_interp = interp1d(f_evol, timesteps)
    f_grid = np.logspace(log_f_LISA_lo, log_f_LISA_hi, n_f_grid)
    t_LISA = t_interp(f_grid) * u.s
    
    return t_LISA
    

def get_LISA_norm(dat):
    m1, m2, e_LIGO = dat
    m1 = m1 * u.Msun
    m2 = m2 * u.Msun
    f_LIGO=10 * u.Hz
    # get time to merger for f_LISA = 1e-4 Hz
    dat = [m1, m2, e_LIGO]
    
    #t_evol = get_t_evol_LISA(dat)
    
    # create timesteps
    timesteps = get_t_evol_from_f(m1, m2, e_LIGO)
    print(timesteps)
    f_orb_evol, ecc_evol = evol.evol_ecc(
        m_1=m1, m_2=m2, f_orb_i=f_LIGO, ecc_i=e_LIGO, timesteps=timesteps,
        t_before=0.0001*u.yr, output_vars=["f_orb", "ecc"], avoid_merger=False)
    if e_LIGO > 0:
        lnJ = -cumtrapz(dg_de(f_orb_evol, ecc_evol), f_orb_evol, initial=0)
        de_deprime = np.exp(lnJ)
    else:
        de_deprime = np.ones(len(f_orb_evol))
    LISA_norm = -1 * dTmerger_df(m1, m2, f_orb_evol, ecc_evol) * de_deprime
        
    return f_orb_evol, ecc_evol, timesteps, LISA_norm


def get_LISA_norm_circular(dat):
    m1, m2, e_LIGO = dat
    m1 = m1 * u.Msun
    m2 = m2 * u.Msun
    f_LIGO = 10 * u.Hz
    # create timesteps
    timesteps = get_t_evol_from_f(m1, m2, e_LIGO, f_LIGO)
    
    f_orb_evol= evol.evol_circ(
        m_1=m1, m_2=m2, f_orb_i=f_LIGO, timesteps=timesteps,
        output_vars=["f_orb"])
    ecc_evol = np.zeros(len(f_orb_evol))

    if e_LIGO > 0:
        lnJ = -cumtrapz(dg_de(f_orb_evol, ecc_evol), f_orb_evol, initial=0)
        de_deprime = np.exp(lnJ)
    else:
        de_deprime = 0.5 * np.ones(len(f_orb_evol))
    LISA_norm = -1 * dTmerger_df(m1, m2, f_orb_evol, ecc_evol) * de_deprime

    return f_orb_evol, ecc_evol, timesteps, LISA_norm
    