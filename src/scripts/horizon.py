import numpy as np
import matplotlib.pyplot as plt
import legwork as lw
import astropy.units as u
import tqdm
from astropy.cosmology import Planck18, z_at_value
from scipy.integrate import trapz
from schwimmbad import MultiPool

e_grid = np.logspace(-3, np.log10(1-0.01), 5)
mass1_range = np.logspace(np.log10(5), np.log10(80), 50)
mass2_range = np.logspace(np.log10(5), np.log10(80), 50)
f_grid = np.logspace(-4, -1.5, 1000)

def get_d_h(e):
    m1_list = []
    m2_list = []
    e_list = []
    f_list = []
    for m1 in mass1_grid:
        for m2 in mass2_grid:
            if m2 < m1:
                             
                e_list.extend(np.zeros(len(f_grid)) * 0)
                m1_list.extend(np.ones(len(f_grid)) * m1)
                m2_list.extend(np.ones(len(f_grid)) * m2)
                f_list.extend(f_grid)
                                
    source = lw.source.Source(m_1=m1_list * u.Msun,
                              m_2=m2_list * u.Msun,
                              ecc=e_list,
                              f_orb=f_list*u.Hz,
                              dist=8 * np.ones(len(e_list)) * u.Mpc,
                              interpolate_g=True)
                                  
    snr = source.get_snr(approximate_R=True, verbose=False)
    
    return snr

with MultiPool(processes=4) as pool:
    values = list(pool.map(get_d_h, list(e_grid)))
