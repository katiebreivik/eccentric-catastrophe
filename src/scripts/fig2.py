import numpy as np
import matplotlib.pyplot as plt
import legwork as lw
import astropy.units as u
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18, z_at_value
from scipy.integrate import trapezoid
import paths
import deepdish as dd
from schwimmbad import MultiPool
import tqdm
import utils
import paths

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 12

# update various fontsizes to match
params = {'figure.figsize': (6,4),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.7 * fs,
          'ytick.labelsize': 0.7 * fs}
plt.rcParams.update(params)


# set up the LISA frequency grid
f_LISA = np.logspace(-1, -5, 150) * u.Hz

# set up the LIGO eccentricity range
e_LIGO = np.logspace(-6, np.log10(0.001), 20)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])


# get the mass, mass ratio, and rate grids
down_samp_fac_q=30
down_samp_fac_m1=15
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac_m1=down_samp_fac_m1, down_samp_fac_q=down_samp_fac_q)

MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

dat_in = list(zip(EE_LIGO.flatten(), FF.flatten(), MM.flatten(), QQ.flatten()*MM.flatten()))

with MultiPool(processes=96) as pool:
    dat_out = list(tqdm.tqdm(pool.imap(utils.get_e_LISA_t_LIGO, dat_in), total=len(dat_in)))
    
EE_LISA, TT_LIGO = zip(*dat_out)

EE_LISA = np.array(EE_LISA).reshape(FF.shape)
TT_LIGO = np.array(TT_LIGO).reshape(FF.shape) * u.yr

np.save(paths.data / 't_merge_run2', TT_LIGO.value)
np.save(paths.data / 'e_LISA_run2', EE_LISA)

def chunk_list(long_list, num_chunks):
    avg = len(long_list) / float(num_chunks)
    chunks = []
    last = 0.0

    while last < len(long_list):
        chunks.append(long_list[int(last):int(last + avg)])
        last += avg

    return chunks


num_chunks = 10

snr_thresh = 12
dat_in = list(zip(MM.flatten(), QQ.flatten(), EE_LISA.flatten(), FF.flatten(), snr_thresh * np.ones(len(MM.flatten()))))

chunked_list = chunk_list(dat_in, num_chunks)
dat_list = []
for ii, chunk in enumerate(chunked_list):
    print('running chunk: ' + str(ii))
    with MultiPool(processes=96) as pool:
        dat_out = list(tqdm.tqdm(pool.imap(utils.get_Vc_Dh, chunk), total=len(chunk)))
        dat_list.extend(dat_out)
DH, VC = zip(*dat_list)
DH = np.array(DH).reshape(QQ.shape) * u.Gpc
VC = np.array(VC).reshape(QQ.shape) * u.Gpc**3

np.save(paths.data / f'comoving_volume_run2', VC.value)
np.save(paths.data / f'horizon_distance_run2', DH.value)

#dT_LIGO_df_LISA = utils.dTmerger_df(MM, QQ*MM, FF, EE_LISA).to(u.yr / u.Hz)

#ind_m_10 = 4
#ind_m_35 = 17
#ind_m_80 = 40
#ind_q_05 = 11
#ind_q_09 = 22
#ind_circ = 0
#ind_ecc_mid = 7
#ind_ecc_high = 15
#
#plt.plot(f_LISA, DH[ind_m_10])
#
print('all done friend!')
