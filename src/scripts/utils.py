from legwork import evol, utils
import legwork as lw
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18, z_at_value

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d, NearestNDInterpolator

def chirp_mass(m1, m2):

    return (m1 * m2)**(5/3) / (m1 + m2)**(1/5)


def dg_de(f, e):
    prefac = 1 / (3 * f)
    numerator = (4477 * e**8 + 99225 * e**6 + 145260 * e**4 + 141472 * e**2 - 29184)
    denominator = (37 * e**4 + 292 * e**2 + 96)**2
    
    return prefac * numerator / denominator
    
    
def dTmerger_df(m1, m2, f, e):
    ecc_fac = (1 - e**2)**(7/2) * (1 + 0.27 * e**10 + 0.33 * e**20 + 0.2 * e**1000)
    dt_df = -5 * np.pi / 48 * c.c**5/(c.G * lw.utils.chirp_mass(m1, m2))**(5/3) * (2 * np.pi * f)**(-11/3)
    
    return ecc_fac * dt_df

def create_timesteps(t_evol=-100*u.yr, nstep_fast=1000, nstep=100):
    t_chunks = np.logspace(0, np.log10(-1*t_evol.to(u.s).value), nstep)
    t_lo = t_chunks[0]
    times = np.logspace(-2, np.log10(t_lo), nstep_fast)
    timesteps = -1 * times * u.s
    for t in t_chunks[1:]:
        times = np.logspace(np.log10(t_lo), np.log10(t), nstep_fast)
        timesteps = np.append(timesteps, -1 * times * u.s)
        t_lo = t
    return timesteps

def get_t_evol_LISA(dat):
    m1, m2, e_LIGO, freq_evol = dat
    f_LIGO=10*u.Hz
    f_LISA=1e-5*u.Hz
    
    if e_LIGO.all() == 0:
        timesteps = create_timesteps(t_evol = -(100000000000)* u.yr, nstep_fast=100 * max(1, int(0.5 * e_LIGO/1e-4)))

        f_evol = evol.evol_circ(
            m_1=m1, m_2=m2, f_orb_i=f_LIGO, timesteps=timesteps,
            output_vars=["f_orb"])
    
    else:
        timesteps = create_timesteps(t_evol = -(10000/e_LIGO)* u.yr, nstep_fast=100 * max(1, int(0.5 * e_LIGO/1e-4)))
        f_evol = evol.evol_ecc(
            m_1=m1, m_2=m2, f_orb_i=f_LIGO, ecc_i=e_LIGO, timesteps=timesteps,
            t_before=0.01*u.yr, output_vars=["f_orb"], avoid_merger=False)
    
    t_interp = interp1d(f_evol, timesteps)
    if f_LISA < min(f_evol):
        print(f_LISA, min(f_evol), m1, e_LIGO)
    t_LISA = t_interp(f_LISA)

    return t_LISA

def get_t_evol_from_f(m1, m2, e_LIGO=1e-5, f_LIGO=10*u.Hz, log_f_LISA_lo=-5, log_f_LISA_hi=-1, n_f_grid=500, plot=False):
    if e_LIGO == 0:
        t_merge = lw.evol.get_t_merge_circ(m_1=m2, m_2=m2, f_orb_i=10**log_f_LISA_lo * u.Hz)
        timesteps = create_timesteps(t_evol = -t_merge, nstep_fast=1000)

        f_evol = evol.evol_circ(
            m_1=m1, m_2=m2, f_orb_i=f_LIGO, timesteps=timesteps,
            output_vars=["f_orb"])
    
    else:
        # Since the eccentricity will grow as we go back in time, we don't know exactly how many 
        # years to go back but a good rule of thumb is to parameterize in terms of e_LIGO
        timesteps = create_timesteps(t_evol = -(10000/e_LIGO)* u.yr, nstep_fast=1000)
        f_evol, ecc_evol = evol.evol_ecc(
            m_1=m1, m_2=m2, f_orb_i=f_LIGO, ecc_i=e_LIGO, timesteps=timesteps,
            t_before=0.001*u.yr, output_vars=["f_orb", "ecc"], avoid_merger=False)
    t_interp = interp1d(f_evol, timesteps)
    #flip because we are integrating backward
    f_grid = np.flip(np.logspace(log_f_LISA_lo, 1, n_f_grid))
    
    t_LISA = t_interp(f_grid) * u.s
    
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(f_evol, ecc_evol, lw=2, label='no interp')

    return t_LISA
    


def get_LISA_params(m1, m2, e_LIGO, a_LIGO, f_LISA_lo=1e-4 * u.Hz, f_LISA_hi=0.1 * u.Hz, n_LISA_step=100):
    if e_LIGO > 0.0:
        e_max = 0.99995
        if e_LIGO >= 1e-3:
            e_grid_steps = 10000
        elif e_LIGO >= 1e-4:
            e_grid_steps = 5000
        else:
            e_grid_steps = 500
        e_grid = np.logspace(np.log10(e_LIGO), np.log10(e_max), e_grid_steps)
        a_grid = lw.utils.get_a_from_ecc(e_grid, lw.utils.c_0(a_i=a_LIGO, ecc_i=e_LIGO))
        log_a_interp = interp1d(a_grid.to(u.Rsun).value, e_grid)
        
        f_LISA_grid = np.logspace(np.log10(f_LISA_hi.value), np.log10(f_LISA_lo.value), n_LISA_step)
        a_LISA_grid = lw.utils.get_a_from_f_orb(m_1=m1, m_2=m2, f_orb=f_LISA_grid * u.Hz)
        e_LISA_grid = log_a_interp(a_LISA_grid.to(u.Rsun).value)
        t_merge_LISA_grid = lw.evol.get_t_merge_ecc(m_1=m1*np.ones(n_LISA_step), m_2=m2*np.ones(n_LISA_step), ecc_i=e_LISA_grid, a_i=a_LISA_grid, exact=True)
        t_merge_LISA_hi = lw.evol.get_t_merge_ecc(m_1=m1, m_2=m2, ecc_i=e_LISA_grid[0], a_i=a_LISA_grid[0], exact=True)
        t_LISA_hi = t_merge_LISA_grid-t_merge_LISA_hi * np.ones(len(t_merge_LISA_grid))
        t_evol_mask = t_LISA_hi < 4 * u.yr
        t_evol = np.ones(len(t_merge_LISA_grid)) * 4 * u.yr
        t_evol[t_evol_mask] = t_LISA_hi[t_evol_mask]
        f_dot_orb_LISA_grid = lw.utils.fn_dot(m_c=lw.utils.chirp_mass(m_1=m1, m_2=m2), f_orb=f_LISA_grid*u.Hz, e=e_LISA_grid, n=1)


    else:
        f_LISA_grid = np.logspace(np.log10(f_LISA_hi.value), np.log10(f_LISA_lo.value), n_LISA_step)
        t_merge_LISA_grid = lw.evol.get_t_merge_circ(m_1=m1, m_2=m2, f_orb_i=f_LISA_grid*u.Hz)
        t_merge_LISA_hi = lw.evol.get_t_merge_ecc(m_1=m1, m_2=m2, ecc_i=0.0, f_orb_i=f_LISA_hi, exact=True)
        t_LISA_hi = t_merge_LISA_grid-t_merge_LISA_hi * np.ones(len(t_merge_LISA_grid))
        t_evol_mask = t_LISA_hi < 4 * u.yr
        t_evol = np.ones(len(t_merge_LISA_grid)) * 4 * u.yr
        t_evol[t_evol_mask] = t_LISA_hi[t_evol_mask]
        f_dot_orb_LISA_grid = lw.utils.fn_dot(m_c=lw.utils.chirp_mass(m_1=m1, m_2=m2), f_orb=f_LISA_grid*u.Hz, e=np.zeros(n_LISA_step), n=1)
        e_LISA_grid = np.zeros(len(f_LISA_grid))

    return [f_LISA_grid*u.Hz, e_LISA_grid, t_merge_LISA_grid, f_dot_orb_LISA_grid, t_evol]

def get_LISA_norm(dat, plot=False):

    m1, m2, e_10 = dat
    m1 = m1 * u.Msun
    m2 = m2 * u.Msun
    f_10=10 * u.Hz
    # create timesteps to get back to the LISA band
    timesteps = get_t_evol_from_f(m1, m2, e_10)
    f_orb_evol, ecc_evol = evol.evol_ecc(
        m_1=m1, m_2=m2, f_orb_i=f_10, ecc_i=e_10, timesteps=timesteps,
        t_before=0.001*u.yr, output_vars=["f_orb", "ecc"], avoid_merger=False)
    
    LISA_mask = (f_orb_evol < 0.1 * u.Hz) & (f_orb_evol > 1e-4 * u.Hz)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(f_orb_evol, ecc_evol, lw=2)
        plt.loglog()
        plt.legend()
        plt.show()
    #if e_10 > 0:
    #    lnJ = cumulative_trapezoid(dg_de(f_orb_evol[LISA_mask],  e_10), -f_orb_evol[LISA_mask], initial=0)
    #    de_deprime = np.exp(lnJ)
    #else:
    #    de_deprime = np.ones(len(f_orb_evol[LISA_mask]))
    
    dT_df = dTmerger_df(m1, m2, f_orb_evol[LISA_mask], ecc_evol[LISA_mask])

    LISA_norm = -1 * dT_df.to(u.s/u.Hz) #* de_deprime
        
    return f_orb_evol[LISA_mask], ecc_evol[LISA_mask], timesteps[LISA_mask], LISA_norm


def get_LISA_norm_circular(dat):
    m1, m2, e_10 = dat
    m1 = m1 * u.Msun
    m2 = m2 * u.Msun
    f_10 = 10 * u.Hz
    # create timesteps from f_10 to f for a range of f in the LISA band
    timesteps = get_t_evol_from_f(m1, m2, e_10, f_10)
    
    f_orb_evol= evol.evol_circ(
        m_1=m1, m_2=m2, f_orb_i=f_10, timesteps=timesteps,
        output_vars=["f_orb"])
    ecc_evol = np.zeros(len(f_orb_evol))

    if e_10 > 0:
        lnJ = cumulative_trapezoid(dg_de(f_orb_evol, e_10), -1 * f_orb_evol, initial=0)
        de_deprime = np.exp(lnJ)
    else:
        de_deprime = 0.5 * np.ones(len(f_orb_evol))
    LISA_norm = -1 * dTmerger_df(m1, m2, f_orb_evol, ecc_evol) * de_deprime

    return f_orb_evol, ecc_evol, timesteps, LISA_norm


def get_horizon(dat_in, snr_thresh=12):
    m1, m2, ecc, forb, t_evol = dat_in
    ind, = np.where(t_evol < 4 * u.yr)
    s = lw.source.Source(m_1=m1,
                         m_2=m2,
                         ecc=ecc,
                         f_orb=forb,
                         dist=8 * np.ones(len(ecc)) * u.Mpc,
                         interpolate_g=False)
    

    snr = s.get_snr()

    for ii in ind:
        s = lw.source.Source(m_1=m1[ii],
                             m_2=m2[ii],
                             ecc=ecc[ii],
                             f_orb=forb[ii],
                             dist=8 * u.Mpc,
                             interpolate_g=False)
        snr_ii = s.get_snr(t_obs = t_evol[ii])
        snr[ii] = snr_ii[0]

    d_horizon = np.ones(len(ecc))* 8 * u.Mpc * snr/snr_thresh

    return d_horizon

def get_D_horizon(m1, m2, e, f, dat_load):
    #Msun, Msun, Hz, Mpc
    M1, M2, E, F, D_horizon = dat_load
    dat_interp = list(zip(M1.flatten(), M2.flatten(), F.flatten(), E.flatten()))
    interp = NearestNDInterpolator(dat_interp, D_horizon.flatten())
    D_H_interp = interp(m1, m2, e, f)
    return D_H_interp * u.Mpc

def get_norms(dat_in, f_LIGO=10*u.Hz):
    m1, m2, e_LIGO, horizon_interp = dat_in
    a_LIGO = lw.utils.get_a_from_f_orb(m_1=m1, m_2=m2, f_orb=f_LIGO) 
    dat_out = get_LISA_params(m1, m2, e_LIGO, a_LIGO)
    f_LISA_grid, e_LISA_grid, t_merge_LISA_grid, f_dot_orb_LISA_grid, t_evol = dat_out
    #dat_in_horizon = [m1, m2, e_LISA_grid, f_LISA_grid, t_evol] = dat_in
    D_h = get_D_horizon(m1, m2, e_LISA_grid, f_LISA_grid, horizon_interp)
    
    redshift = np.ones(len(D_h)) * 1e-8
    redshift[D_h > 10 * u.kpc] = z_at_value(Planck18.luminosity_distance, D_h[D_h > 10 * u.kpc])
    V_c = 4/3 * np.pi * D_h**3  

    V_c[D_h > 10 * u.kpc] = Planck18.comoving_volume(z=redshift[D_h > 10 * u.kpc])    

    dT_df = dTmerger_df(m1, m2, f_LISA_grid, e_LISA_grid)
    
    dat_out = [-1 * dT_df.to(u.s/u.Hz), e_LISA_grid, f_LISA_grid, D_h, V_c]
    return dat_out

    