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

def get_LIGO_rate(down_samp_fac=1):
    # this is lifted ~exactly~ from the GWTC-3 tutorial
    mass_PP_path = paths.data / "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
    with open(mass_PP_path, 'r') as _data:
        _data = dd.io.load(mass_PP_path)
    
    #import pdb
    #pdb.set_trace()
    dN_dm1dqdVcdt = _data['ppd'].T
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    M1, Q = np.meshgrid(mass_1, mass_ratio, indexing='ij')
    
    if down_samp_fac > 1:
        mass_1 = mass_1[::down_samp_fac]
        mass_ratio = mass_ratio[::down_samp_fac]
        M1 = M1[::down_samp_fac, ::down_samp_fac]
        Q = Q[::down_samp_fac, ::down_samp_fac]
        dN_dm1dqdVcdt = dN_dm1dqdVcdt[::down_samp_fac, ::down_samp_fac]

    return mass_1*u.Msun, mass_ratio, M1*u.Msun, Q, dN_dm1dqdVcdt*u.Msun**(-1) * u.Gpc**(-3) * u.yr**(-1)

def get_horizon_and_chirp(dat_in):
    m1, q, f = dat_in
    m2 = m1 * q
    mc = lw.utils.chirp_mass(m1, m2)

    s = lw.source.Source(m_1=m1 * np.ones(len(f)),
                         m_2=m2 * np.ones(len(f)),
                         ecc=np.zeros(len(f)),
                         f_orb=f,
                         dist=8 * np.ones(len(f)) * u.Mpc,
                         interpolate_g=False,
                         gw_lum_tol=0.001)
    
    
    snr = s.get_snr(t_obs = 8 * u.yr, approximate_R=True, verbose=False)
    d_h_1 = snr * 8 * u.Mpc
    d_h_7 = snr / 7 * 8 * u.Mpc    
    d_h_12 = snr / 12 * 8 * u.Mpc
    
    
    d_h_list = [d_h_1.to(u.Gpc), d_h_7.to(u.Gpc), d_h_12.to(u.Gpc)]
    V_h_list = []
    for d_h in d_h_list:
        V_h = 4/3 * np.pi * d_h**3

        d_mask = d_h > 100 * u.kpc
        redshift = np.ones(len(d_h)) * 1e-8
        redshift[d_mask] = z_at_value(Planck18.luminosity_distance, d_h[d_mask])
        V_h[d_mask] = Planck18.comoving_volume(z=redshift[d_mask])
        
        V_h_list.append(V_h.to(u.Gpc**3))

    f_dot = lw.utils.fn_dot(m_c=mc, e=np.zeros(len(f)), n=1, f_orb=f)
    
    d_h_list = [d_h.value for d_h in d_h_list]
    V_h_list = [V_h.value for V_h in V_h_list]
    
    return d_h_list, V_h_list, f_dot.to(u.Hz/u.s).value


# set up the LISA frequency grid
f_LISA_grid = np.logspace(-1, -5, 500) * u.Hz


# get the mass, mass ratio, and rate grids
down_samp_fac=15
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = get_LIGO_rate(down_samp_fac=down_samp_fac)

# run on 98 processors
nproc=128

# loop over the mass and mass ratio grid to get the horizon distance
dat_in = []
for m1, q in zip(M1.flatten(), Q.flatten()):
    dat_in.append([m1, q, f_LISA_grid])
    
with MultiPool(processes=nproc) as pool:
    dat_out = list(tqdm.tqdm(pool.imap(get_horizon_and_chirp, dat_in), total=len(dat_in)))

D_H_1 = []
D_H_7 = []
D_H_12 = []
V_H_1 = []
V_H_7 = []
V_H_12 = []
F_DOT = []
for d in dat_out:
    d_h, V_h, f_dot = d

    D_H_1.append(d_h[0])
    D_H_7.append(d_h[1])
    D_H_12.append(d_h[2])

    V_H_1.append(V_h[0])
    V_H_7.append(V_h[1])
    V_H_12.append(V_h[2])
    
    F_DOT.append(f_dot)
    
D_H_1 = np.reshape(D_H_1, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Gpc    
V_H_1 = np.reshape(V_H_1, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Gpc**3    
D_H_7 = np.reshape(D_H_7, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Gpc    
V_H_7 = np.reshape(V_H_7, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Gpc**3    
D_H_12 = np.reshape(D_H_12, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Gpc    
V_H_12 = np.reshape(V_H_12, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Gpc**3    
F_DOT = np.reshape(F_DOT, (len(mass_1), len(mass_ratio), len(f_LISA_grid))) * u.Hz / u.s

# get the rates
dN_dm1_list = []
N_LISA_obs_list = []
for v in [V_H_1, V_H_7, V_H_12]:
    dN_dm1 = np.zeros(len(mass_1)) / u.Msun
    
    for ii, m in enumerate(mass_1):
        dN_dm1dq = np.zeros(len(mass_ratio)) / u.Msun
        for jj, q in enumerate(mass_ratio):
            dN_dm1dq[jj] = trapezoid(dN_dm1dqdVcdt[ii,jj] / -F_DOT[ii,jj,:] * v[ii,jj,:], f_LISA_grid).to(1/u.Msun)
        dN_dm1[ii] = trapezoid(dN_dm1dq, mass_ratio)
    N_LISA_obs=trapezoid(dN_dm1, mass_1)
    print(f'The number of LISA sources in total is: {np.round(N_LISA_obs, 2)}')

    dN_dm1_list.append(dN_dm1)    
    N_LISA_obs_list.append(N_LISA_obs)
dN_dm1_LIGO = trapezoid(dN_dm1dqdVcdt, mass_ratio, axis=1)                        


fig, ax1 = plt.subplots(figsize=(6,4))

# Plot data on the first y-axis
color = 'black'
ax1.set_xlabel(r'$M_{BH,1}$ [$M_{\odot}$]', size=16)
ax1.set_ylabel(r'$dN_{\rm{LISA}}/dM_{BH,1}$ [$M_{\odot}^{-1}$]', color=color, size=16)
for dN_dm1, ls, snr_r, N_LISA_obs in zip(dN_dm1_list, ['-', '--', '-.'], [1,7,12], N_LISA_obs_list):
    ax1.plot(mass_1, dN_dm1.value, color=color, label=f'SNR > {snr_r}', ls=ls)
    
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.set_yscale('log')
ax1.set_ylim(1e-5, 3e2)
# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot data on the second y-axis
color = 'navy'
ax2.set_ylabel(r'$dN_{\rm{intrinsic}}/dM_{BH,1}$ [$M_{\odot}^{-1}$]', color=color, size=16)
ax2.plot(mass_1, dN_dm1_LIGO, color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.set_yscale('log')
ax2.set_ylim(5e-5, 7)
plt.minorticks_on()
ax1.set_xlim(0, 95)
ax1.legend(prop={"size":12}, ncol=3, loc=(0, 1.01))


#plt.minorticks_on()
fig.tight_layout()
fig.savefig(paths.figures / 'fig1_8yr.png', facecolor='white', dpi=100)
    