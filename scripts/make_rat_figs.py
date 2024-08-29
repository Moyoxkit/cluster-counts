import json
import pickle
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import dblquad, quad
from scipy.interpolate import (
    interp1d,
    LinearNDInterpolator,
    NearestNDInterpolator,
    griddata,
)
from scipy.special import erf
import MiraTitanHMFemulator
import pandas as pd
import unyt
from matplotlib.pyplot import style
from matplotlib import rc

m_nu = [0.02, 0.02, 0.02] * u.eV
astropy_cosmology = FlatLambdaCDM(
    H0=68.1, Om0=0.3046, m_nu=m_nu, Ob0=0.0486, Tcmb0=2.725
)

masses = np.logspace(12, 15.5, 36)


def get_hmf_from_cat(dictionary_path):
    with open(dictionary_path, "rb") as f:
        dictionary_with_hmf = pickle.load(f)
    all_hmfs = np.zeros((61, 36))
    for i in range(41):
        for k in range(36):
            all_hmfs[i, k] = dictionary_with_hmf[str(i)][str(k)]["dn_dlog10m"]

    return all_hmfs


def get_hmf_from_cat_56(dictionary_path):
    with open(dictionary_path, "rb") as f:
        dictionary_with_hmf = pickle.load(f)
    all_hmfs = np.zeros((41, 36))
    for i in range(1):
        for k in range(36):
            all_hmfs[i, k] = dictionary_with_hmf[str(i)][str(k)]["dn_dlog10m"]

    return all_hmfs


def get_scaling_from_cat(dictionary_path):
    with open(dictionary_path, "rb") as f:
        dictionary_with_hmf = pickle.load(f)
    all_meds = np.zeros((61, 36))
    all_scatter = np.zeros((61, 36, 2))
    for i in range(41):
        for k in range(36):
            all_meds[i, k] = dictionary_with_hmf[str(i)][str(k)]["median"]
            bins = dictionary_with_hmf[str(i)][str(k)]["bins"]
            hist = dictionary_with_hmf[str(i)][str(k)]["hist"]
            cumulative = 1 - np.cumsum(hist) / np.sum(hist)
            get_percent = interp1d(cumulative, bins)
            all_scatter[i, k, 0] = get_percent(0.16)
            all_scatter[i, k, 1] = get_percent(0.84)
    return all_meds, all_scatter


style.use("../colibre-stylesheer.mplsht")
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["font.size"] = 10

dict_folder = "../make_hbt_pickles/"

FLAMINGO_colors = [
    "#117733",
    "#332288",
    "#DDCC77",
    "#CC6677",
    "#abd0e6",
    "#6aaed6",
    "#3787c0",
    "#105ba4",
    "#FF8C40",
    "#CC4314",
    "#7EFF4B",
    "#55E18E",
    "#44AA99",
    "#999933",
    "#AA4499",
    "#882255",
]
FLAMINGO_labels = [
    r"L1$\_$m9",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    "Jet",
    r"Jet$\_$fgas$-4\sigma$",
    "5p6_DMO",
    "L1_DMO",
    "Tinker (2010)",
    "Bocquet (2016)",
    "MiraTitanEmulator (Bocquet 2020)",
]

FLAMINGO_plots = {name: color for name, color in zip(FLAMINGO_labels, FLAMINGO_colors)}

FLAMINGO_colors = np.array(FLAMINGO_colors)[np.array([0, 4, 5, 6, 7, 10, 11])]

med_catalogues = np.array(
    [
        dict_folder + "hist_and_med_L1000N1800_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_WEAK_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONG_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONGER_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONGEST_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_JETS_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONG_JETS_Y.pkl",
        dict_folder + "hist_and_med_L5600N5040DMO.pkl",
        dict_folder + "hist_and_med_L1000N1800DMO.pkl",
        "HMF_cats/HMF_Tink10.pkl",
        "HMF_cats/HMF_Bocquet500cDMOnly.pkl",
        "HMF_cats/HMFMiratitanEmu.pkl",
    ]
)

extra_colors = np.array([cycle[3], cycle[4], cycle[0], cycle[1], cycle[2]])
FLAMINGO_colors = np.append(FLAMINGO_colors, extra_colors)


to_devide_by = get_hmf_from_cat_56(med_catalogues[7])[0, :]

fig = plt.figure(figsize=(4, 3.321))
ax2 = fig.add_subplot(111)

plt.axhline(1, color="black", ls=":")
plt.axhline(0.9, color="grey", ls=":")
plt.axhline(1.1, color="grey", ls=":")

plot2_indices = [9, 10, 11]

for index, cat_loc in enumerate(med_catalogues[plot2_indices]):
    print(cat_loc)
    ax2.plot(
        masses,
        get_hmf_from_cat(cat_loc)[0, :] / to_devide_by,
        color=FLAMINGO_colors[plot2_indices[index]],
        label=FLAMINGO_labels[plot2_indices[index]],
    )

N = to_devide_by * 1000**3 * 0.1
ax2.set_xscale("log")
ax2.legend(ncol=1, fontsize=9, fancybox=True, frameon=True)
ax2.set_ylabel("HMF/5p6_DMO")
ax2.set_xlabel(r"$M_{\rm 500c}~[\rm{M}_{\odot}]$")
ax2.set_ylim(0.5, 1.5)
ax2.set_xlim(1e13, 5e15)
plt.axvline(2e14, color="grey", ls=":")
plt.tight_layout()
plt.savefig("figures/HMF_rats.pdf")


def HMF_ratio_maker(DMO_path, HYDRO_path):
    with open(DMO_path, "rb") as f:
        dmo_dictionary_with_hmf = pickle.load(f)
    with open(HYDRO_path, "rb") as f:
        hyd_dictionary_with_hmf = pickle.load(f)

    dmo_hmfs = np.zeros((61, 36))
    hyd_hmfs = np.zeros((61, 36))
    for i in range(41):
        for k in range(36):
            dmo_val = dmo_dictionary_with_hmf[str(i)][str(k)]["dn_dlog10m"]
            hyd_val = hyd_dictionary_with_hmf[str(i)][str(k)]["dn_dlog10m"]
            if hyd_val == 0 or dmo_val == 0:
                dmo_hmfs[i, k] = 1
                hyd_hmfs[i, k] = 1
            else:
                dmo_hmfs[i, k] = dmo_val
                hyd_hmfs[i, k] = hyd_val

    return hyd_hmfs / dmo_hmfs


fig = plt.figure(figsize=(4, 3.321))
ax2 = fig.add_subplot(111)

for k in range(7):
    rats = HMF_ratio_maker(med_catalogues[8], med_catalogues[k])
    plt.plot(masses, rats[0, :], color=FLAMINGO_colors[k], label=FLAMINGO_labels[k])


plt.legend(
    fontsize=9, ncol=2, title="Baryonic modification", fancybox=True, frameon=True
)
plt.ylim(0.6, 1.4)
plt.axhline(1, color="black", ls=":")
plt.axhline(0.9, color="grey", ls=":")
plt.axhline(1.1, color="grey", ls=":")
plt.xlim(1e13, 5e15)
plt.ylabel("HMF/L1_DMO")
plt.xlabel(r"$M_{\rm 500c}~[\rm{M}_{\odot}]$")
plt.axvline(2e14, color="grey", ls=":")
plt.xscale("log")
plt.savefig("figures/HMF_bar_rats.pdf")

dividemeds, standard_scatter = get_scaling_from_cat(med_catalogues[0])

to_devide_by = get_hmf_from_cat_56(med_catalogues[7])[0, :]

N = to_devide_by * 1000**3 * 0.1
to_fit_too = dividemeds[:, 20:-2].flatten()
mask = np.logical_or((np.isnan(to_fit_too)), (to_fit_too <= 0))


def get_cy_for_masses(masses, Ystar, alpha, beta, z):
    return Ystar * (
        (astropy_cosmology.H(z) / astropy_cosmology.H0) ** (beta)
        * (astropy_cosmology.H(z) / (70 * u.km / (u.s * u.Mpc))) ** (alpha - 2)
        * (0.743 * masses / 6e14) ** alpha
        * 1e-4
    )


def make_cy_for_all_masses(params):
    Ystar, alpha, beta = params
    used_masses = masses[20:-2]
    all_points = np.zeros((61, len(used_masses)))
    z = 0
    for i in range(61):
        all_points[i] = Ystar * (
            (astropy_cosmology.H(z) / astropy_cosmology.H0) ** (beta)
            * (astropy_cosmology.H(z) / (70 * u.km / (u.s * u.Mpc))) ** (alpha - 2)
            * (0.743 * used_masses / 6e14) ** alpha
            * 1e-4
        )
        z += 0.05
    return np.abs(np.mean(all_points.flatten()[~mask] - to_fit_too.flatten()[~mask]))


print(make_cy_for_all_masses([10 ** (-0.095), 1.66, 0.89]))
import scipy.optimize as optimize

result = optimize.minimize(make_cy_for_all_masses, [10 ** (-0.095), 1.66, 0.89])
print(result.x)
np.save("fit_params", result.x)
print(np.log10(result.x[0]))

# scaling = get_cy_for_masses(masses, best_fit[0][0], best_fit[0][1], 0.93, 0.5)

dividemeds, standard_scatter = get_scaling_from_cat(med_catalogues[0])
fig = plt.figure(figsize=(4, 3.321))
ax2 = fig.add_subplot(111)

max_z = 41
col_index = np.linspace(0, 1, max_z)

for k in range(41):
    scaling = get_cy_for_masses(
        masses, 10 ** (-0.09858391229313404), 1.65962885, 0.88851587, k * 0.05
    )
    plt.plot(
        masses, scaling / dividemeds[k, :], color=plt.cm.plasma(col_index[k]), alpha=0.6
    )

import matplotlib as mpl

cmap = mpl.cm.plasma

plt.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 2)),
    cmap=cmap,
    label=r"$z$",
    ax=ax2,
)
plt.xscale("log")
plt.xlim(1e13, 5e15)
plt.ylim(0.6, 1.4)
plt.axhline(1, color="black", ls=":")
plt.xlabel(r"$M_{\rm 500c}~[\rm{M}_{\odot}]$")
plt.ylabel(r"$Y_{\rm 500c, PL}/Y_{\rm 500c,L1\_m9}$")
plt.legend(fontsize=8)
plt.axhline(0.9, color="grey", ls=":")
plt.axhline(1.1, color="grey", ls=":")
plt.axvline(2e14, color="grey", ls=":")
plt.savefig("figures/scal_PL_rats.pdf")


dividemeds, standard_scatter = get_scaling_from_cat(med_catalogues[0])
fig = plt.figure(figsize=(4, 3.321))
ax2 = fig.add_subplot(111)

ind_to_plot = range(7)
for k in ind_to_plot:
    if k == 0:
        continue
    meds, scatter = get_scaling_from_cat(med_catalogues[k])
    plt.plot(
        masses,
        meds[10, :] / dividemeds[10, :],
        color=FLAMINGO_colors[k],
        label=FLAMINGO_labels[k],
    )

# plt.plot(masses, scaling / dividemeds[10, :], color=cycle[4], label="PL")
plt.xscale("log")
plt.xlim(1e13, 5e15)
plt.ylim(0.6, 1.4)
plt.axhline(1, color="black", ls=":")
plt.xlabel(r"$M_{\rm 500c}~[\rm{M}_{\odot}]$")
plt.ylabel(r"$Y_{\rm 500c}/Y_{\rm 500c,L1\_m9}$")
plt.legend(
    fontsize=9,
    ncol=2,
    title="Scaling relation from simulation",
    fancybox=True,
    frameon=True,
)
plt.axhline(0.9, color="grey", ls=":")
plt.axhline(1.1, color="grey", ls=":")
plt.axvline(2e14, color="grey", ls=":")
plt.savefig("figures/scal_bar_rats.pdf")
