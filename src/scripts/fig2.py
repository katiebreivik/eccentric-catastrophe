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
import cmasher as cmr

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
e_LIGO = np.logspace(-6, np.log10(0.0005), 10)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])


# get the mass, mass ratio, and rate grids
down_samp_fac=25
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac=down_samp_fac)

mass_1 = mass_1
mass_ratio = mass_ratio
MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

#dat_in = list(zip(EE_LIGO.flatten(), FF.flatten(), MM.flatten(), QQ.flatten()*MM.flatten()))
#
#with MultiPool(processes=96) as pool:
#    dat_out = list(tqdm.tqdm(pool.imap(utils.get_e_LISA_t_LIGO, dat_in), total=len(dat_in)))
#    
#EE_LISA, TT_LIGO = zip(*dat_out)
#
#EE_LISA = np.array(EE_LISA).reshape(FF.shape)
#TT_LIGO = np.array(TT_LIGO).reshape(FF.shape) * u.yr
#
#np.save(paths.data / 't_merge', TT_LIGO.value)
#np.save(paths.data / 'e_LISA', EE_LISA)

#def chunk_list(long_list, num_chunks):
#    avg = len(long_list) / float(num_chunks)
#    chunks = []
#    last = 0.0
#
#    while last < len(long_list):
#        chunks.append(long_list[int(last):int(last + avg)])
#        last += avg
#
#    return chunks
#
#
#num_chunks = 10
#
#snr_thresh = 12
#dat_in = list(zip(MM.flatten(), QQ.flatten(), EE_LISA.flatten(), FF.flatten(), snr_thresh * np.ones(len(MM.flatten()))))
#
#chunked_list = chunk_list(dat_in, num_chunks)
#dat_list = []
#for ii, chunk in enumerate(chunked_list):
#    print('running chunk: ' + str(ii))
#    with MultiPool(processes=96) as pool:
#        dat_out = list(tqdm.tqdm(pool.imap(utils.get_Vc_Dh, chunk), total=len(chunk)))
#        dat_list.extend(dat_out)
#DH, VC = zip(*dat_list)
#DH = np.array(DH).reshape(QQ.shape) * u.Gpc
#VC = np.array(VC).reshape(QQ.shape) * u.Gpc**3
#
#np.save(paths.data / f'comoving_volume_{ii}', VC.value)
#np.save(paths.data / f'horizon_distance_{ii}', DH.value)
EE_LISA = np.load(paths.data / 'e_LISA.npy')
TT_LISA = np.load(paths.data / 't_merge.npy') * u.Gyr
VC = np.load(paths.data / 'comoving_volume.npy') * u.Gpc**3
DH = np.load(paths.data / 'horizon_distance.npy') * u.Gpc 

dT_LIGO_df_LISA = utils.dTmerger_df(MM, QQ*MM, FF, EE_LISA).to(u.yr / u.Hz)

snr_thresh_data = 12
snr_thresh_list = [1, 7, 12]
DH = DH * snr_thresh_data / snr_thresh_new
VC = 4/3 * np.pi * DH**3

cs = cmr.take_cmap_colors('cmr.dusk', len(mass_1), cmap_range=(0.15, 0.9), return_fmt='hex')

ind_m_10 = 3
ind_m_35 = 14
ind_m_80 = 32
ind_q05 = 9
ind_q09 = 18
ind_circ = 0
ind_ecc_mid = 4
ind_ecc_high = 10

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,3))
ax1.plot(f_LISA.value, DH[ind_m_10,ind_q09,ind_circ,:].value, ls='-', color=cs[ind_m_10], zorder=10, label=r'$M_1=10\,M_{\odot}$')
ax2.plot(f_LISA.value, DH[ind_m_10,ind_q09,ind_ecc_mid,:].value, ls='--', color=cs[ind_m_10], zorder=10, label=r'$e_{\rm{LIGO}}=10^{-5}$')
ax3.plot(f_LISA.value, DH[ind_m_10,ind_q09,ind_ecc_high,:].value, ls=':', color=cs[ind_m_10], zorder=10, label=r'$e_{\rm{LIGO}}=10^{-3}$')
#plt.fill_between(f_LISA.value, DH[ind_m_10,ind_q09,ind_circ,:].value, DH[ind_m_10,ind_q09,ind_ecc_high,:].value, color=cs[ind_m_10], alpha=0.5, zorder=10)

ax1.plot(f_LISA.value, DH[ind_m_35,ind_q09,ind_circ,:].value, ls='-', color=cs[ind_m_35], zorder=5, label=r'$M_1=35\,M_{\odot}$')
ax2.plot(f_LISA.value, DH[ind_m_35,ind_q09,ind_ecc_mid,:].value, ls='--', color=cs[ind_m_35], zorder=5)
ax3.plot(f_LISA.value, DH[ind_m_35,ind_q09,ind_ecc_high,:].value, ls=':', color=cs[ind_m_35], zorder=5)
#plt.fill_between(f_LISA.value, DH[ind_m_35,ind_q09,ind_circ,:].value, DH[ind_m_35,ind_q09,ind_ecc_high,:].value, color=cs[ind_m_35], alpha=0.5, zorder=5)


ax1.plot(f_LISA.value, DH[ind_m_80,ind_q09,ind_circ,:].value, ls='-', color=cs[ind_m_80], label=r'$M_1=80\,M_{\odot}$')
ax2.plot(f_LISA.value, DH[ind_m_80,ind_q09,ind_ecc_mid,:].value, ls='--', color=cs[ind_m_80])
ax3.plot(f_LISA.value, DH[ind_m_80,ind_q09,ind_ecc_high,:].value, ls=':', color=cs[ind_m_80])
#plt.fill_between(f_LISA.value, DH[ind_m_80,ind_q09,ind_circ,:].value, DH[ind_m_80,ind_q09,ind_ecc_high,:].value, color=cs[ind_m_80], alpha=0.5)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(1e-5, 6)
ax1.set_xlim(1e-5, 1e-1)
ax1.set_xlabel(r'$f_{\rm{LISA}}$ [Hz]')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(1e-5, 6)
ax2.set_xlim(1e-5, 1e-1)
ax2.set_xlabel(r'$f_{\rm{LISA}}$ [Hz]')

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(1e-5, 6)
ax3.set_xlim(1e-5, 1e-1)
ax3.set_xlabel(r'$f_{\rm{LISA}}$ [Hz]')

ax1.legend(loc='upper left', frameon=False, prop={'size':11})
ax2.legend(loc='upper left', frameon=False, prop={'size':11})
ax3.legend(loc='upper left', frameon=False, prop={'size':11})

ax1.set_ylabel('horizon distance [Gpc]')

fig.tight_layout()
fig.savefig(paths.figures / 'fig2.png', facecolor='white', dpi=100)
