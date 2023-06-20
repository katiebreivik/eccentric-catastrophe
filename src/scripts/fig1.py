import numpy as np
import matplotlib.pyplot as plt
import legwork as lw
import astropy.units as u
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18, z_at_value
from scipy.integrate import trapz
import paths

def ligo_rate(m1):
    dat = np.array([[3.705799151343708, 0.001087789470121345],
                   [4.384724186704389, 0.00984816875074369],
                   [5.063649222065067, 0.06979974252228799],
                   [5.827439886845831, 0.41173514594201527],
                   [6.506364922206512, 1.3579705933006465],
                   [6.845827439886847, 2.148948034692836],
                   [7.77934936350778, 2.7449738151212433],
                   [8.543140028288544, 2.6218307403757986],
                   [9.561527581329564, 2.0525434471508692],
                   [11.173974540311175, 1.2388629239937763],
                   [12.701555869872706, 0.7828664968878465],
                   [14.398868458274404, 0.4947116747780942],
                   [16.859971711456865, 0.2895969742197884],
                   [19.66053748231967, 0.17748817964452962],
                   [22.206506364922213, 0.12773570001722281],
                   [24.837340876944843, 0.10389898279212807],
                   [27.722772277227726, 0.1087789470121345],
                   [30.183875530410184, 0.13070104796093673],
                   [32.729844413012735, 0.16441704701060267],
                   [34.85148514851486, 0.16695189854274867],
                   [37.397454031117405, 0.12107555776371784],
                   [39.26449787835927, 0.08010405199404155],
                   [41.30127298444131, 0.049851062445855264],
                   [43.592644978783596, 0.029631988560550687],
                   [45.629420084865636, 0.018440841322693136],
                   [48.0905233380481, 0.011832859313068754],
                   [50.891089108910904, 0.007949361111716631],
                   [53.77652050919379, 0.005764973856945108],
                   [57.25601131541727, 0.0043438393396653925],
                   [61.923620933521946, 0.0032730313574784275],
                   [66.67609618104669, 0.0024851284269805634],
                   [70.66478076379069, 0.002068305171949823],
                   [74.82319660537483, 0.0016952583040389245],
                   [78.72701555869875, 0.0013476220436441713],
                   [81.27298444130128, 0.0010389898279212807]])
    
    mass = dat[:,0]
    rate = dat[:,1]
    interp_rate = interp1d(mass, rate)
    
    return interp_rate(m1)

n_grid = 25

# set up the grids
f = np.logspace(-1, -5, 100) * u.Hz
masses = np.arange(5, 80.2, 0.2)
m_c = lw.utils.chirp_mass(masses, masses)

# set up bins
delta_m = np.mean(masses[1:] - masses[:-1])/2
mass_bins = np.arange(min(masses) - delta_m, max(masses) + 3 * delta_m, 2*delta_m)
masses = masses * u.Msun
mass_bins = mass_bins * u.Msun

# get the meshgrid
F, MASS = np.meshgrid(f, masses)
MC = lw.utils.chirp_mass(MASS, MASS)
RATE = ligo_rate(MASS.flatten().value)
RATE = RATE.reshape(MC.shape) * u.Gpc**(-3) * u.yr**(-1) * u.Msun**(-1)

# get the horizon distance
source = lw.source.Source(m_1=MASS.flatten(),
                          m_2=MASS.flatten(),
                          ecc=np.zeros(len(F.flatten())),
                          f_orb=F.flatten(),
                          dist=8 * np.ones(len(F.flatten())) * u.Mpc,
                          interpolate_g=False,
                          n_proc=1)
snr = source.get_snr(approximate_R=False, verbose=True)
SNR_resolve = 12
D_h = snr/SNR_resolve * 8 * u.Mpc
redshift = np.ones(len(D_h)) * 1e-8
redshift[D_h > 0.0001 * u.Mpc] = z_at_value(Planck18.luminosity_distance, D_h[D_h > 0.0001 * u.Mpc])
horizon_comoving_volume = Planck18.comoving_volume(z=redshift)
horizon_comoving_volume = horizon_comoving_volume.reshape(RATE.shape)
D_h = D_h.reshape(RATE.shape)

# calculate the chirp
f_dot = lw.utils.fn_dot(m_c = MC.flatten(), e = np.zeros(len(MC.flatten())), n=2, f_orb=F.flatten())
f_dot = f_dot.reshape(F.shape)

# get the rates
N_per_mass = np.zeros(len(masses)) * u.Msun**(-1)
for ii, m in enumerate(masses):
    N_per_mass[ii] = trapz(RATE[ii,:] / f_dot[ii,:] * horizon_comoving_volume[ii,:], -f)


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(masses, N_per_mass, lw=3, label=r"LISA")
ax.plot(masses, ligo_rate(masses)/45, lw=2, label=r"LIGO")
plt.legend(prop={"size":14})
plt.tick_params('both', labelsize=12)
plt.minorticks_on()
ax.set_xlabel('mass [Msun]', size=16)
ax.set_ylabel('p(M) [M$_{\odot}^{-1}$]', size=16)
plt.tight_layout()
plt.savefig(paths.figures / 'fig1.png', facecolor='white', dpi=100)