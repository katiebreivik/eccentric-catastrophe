import numpy as np
import legwork as lw
import astropy.units as u
import argparse

def save_horizon_grid(fname='horizon_dat'):
    '''Calculate the LISA horizon distance on a 
    regular grid of masses, eccentricities, and frequencies

    Parameters
    ----------
    fname : 'str'
        Name of file to save horizon distance and grid array

    Returns
    -------
    None
    '''
    
    e_grid = np.array([0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85])
    mass_grid = np.logspace(np.log10(5), np.log10(80), 15)
    q_grid = np.linspace(0.1, 1, 15)
    f_grid = np.logspace(-4, -1.0, 100)
    
    M1, Q, E, F = np.meshgrid(mass_grid, q_grid, e_grid, f_grid)
    M2 = M1 * Q   
    
    source = lw.source.Source(m_1=M1.flatten() * u.Msun,
                              m_2=M2.flatten() * u.Msun,
                              ecc=E.flatten(),
                              f_orb=F.flatten()*u.Hz,
                              dist=8 * np.ones(len(E.flatten())) * u.Mpc,
                              interpolate_g=True)
                                      
    snr = source.get_snr(approximate_R=True, verbose=False)
    
    snr_thresh = 12.0
    D_horizon = snr / snr_thresh * 8 * u.Mpc
    D_horizon = np.reshape(D_horizon, M1.shape)
    
    dat_all = np.array([M1, M2, E, F, D_horizon.value])
    np.save(fname, dat_all, allow_pickle=True)

    return

if __name__=='__main__':

    # set up the argparser
    parser = argparse.ArgumentParser(
                    prog='horizon',
                    description='Calculate the horizon distance for LISA on a grid')

    parser.add_argument('-f', '--fname')      # option that takes a value

    args = parser.parse_args()

    _= save_horizon()