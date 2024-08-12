import numpy as np
from tqdm import tqdm
import h5py as h5
import pickle
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model

redshifts = np.linspace(0, 2, 41)
FLAM_info = median_and_scatter_model(
    "../make_hbt_pickles/hist_and_med_L1000N1800_Y.pkl"
)
SZ_counts = ccm(FLAM_info, 1e-4, mira_titan=True)
mass_limits = np.logspace(12, 15.5, 36)

output_dict = {}
for snapshot_ind in tqdm(range(41)):  # 61 is z = 3
    dict_for_snap = {}
    dict_for_snap["z"] = redshifts[snapshot_ind]
    all_number_dens = SZ_counts.halo_mass_function(
        mass_limits, redshifts[snapshot_ind], mira_titan=True
    )
    mass_ind = 0
    for mass_limit in tqdm(mass_limits, leave=False):
        hist_dict = {}
        hist_dict["Mass_cut"] = mass_limit
        hist_dict["dn_dlog10m"] = all_number_dens[mass_ind]
        dict_for_snap[str(mass_ind)] = hist_dict
        mass_ind += 1
        output_dict[str(snapshot_ind)] = dict_for_snap

with open("HMF_cats/HMFMiratitanEmu.pkl", "wb") as f:
    pickle.dump(output_dict, f)

FLAM_info = median_and_scatter_model(
    "../make_hbt_pickles/hist_and_med_L1000N1800_Y.pkl"
)
SZ_counts = ccm(FLAM_info, 1e-4, fit_model="Tinker10")
mass_limits = np.logspace(12, 15.5, 36)

output_dict = {}
for snapshot_ind in tqdm(range(41)):  # 61 is z = 3
    dict_for_snap = {}
    dict_for_snap["z"] = redshifts[snapshot_ind]
    all_number_dens = SZ_counts.halo_mass_function(
        mass_limits, redshifts[snapshot_ind], mira_titan=False
    )
    mass_ind = 0
    for mass_limit in tqdm(mass_limits, leave=False):
        hist_dict = {}
        hist_dict["Mass_cut"] = mass_limit
        hist_dict["dn_dlog10m"] = all_number_dens[mass_ind]
        dict_for_snap[str(mass_ind)] = hist_dict
        mass_ind += 1
        output_dict[str(snapshot_ind)] = dict_for_snap


with open("HMF_cats/HMF_Tink10.pkl", "wb") as f:
    pickle.dump(output_dict, f)

FLAM_info = median_and_scatter_model(
    "../make_hbt_pickles/hist_and_med_L1000N1800_Y.pkl"
)
SZ_counts = ccm(FLAM_info, 1e-4, fit_model="Bocquet500cDMOnly")
mass_limits = np.logspace(12, 15.5, 36)

output_dict = {}
for snapshot_ind in tqdm(range(41)):  # 61 is z = 3
    dict_for_snap = {}
    dict_for_snap["z"] = redshifts[snapshot_ind]
    all_number_dens = SZ_counts.halo_mass_function(
        mass_limits, redshifts[snapshot_ind], mira_titan=False
    )
    mass_ind = 0
    for mass_limit in tqdm(mass_limits, leave=False):
        hist_dict = {}
        hist_dict["Mass_cut"] = mass_limit
        hist_dict["dn_dlog10m"] = all_number_dens[mass_ind]
        dict_for_snap[str(mass_ind)] = hist_dict
        mass_ind += 1
        output_dict[str(snapshot_ind)] = dict_for_snap

with open("HMF_cats/HMF_Bocquet500cDMOnly.pkl", "wb") as f:
    pickle.dump(output_dict, f)
