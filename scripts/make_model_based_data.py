import pandas as pd
import numpy as np
import unyt
from tqdm import tqdm
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model

redshift_edges = np.linspace(0, 2.0, 21)

SZ_cut = [1e-4, 3e-5, 1e-5]

dict_folder = "../make_hbt_pickles/"

selec_file = dict_folder + "hist_and_med_L1000N1800_Y.pkl"
hmf_file = dict_folder + "hist_and_med_L5600N5040DMO.pkl"

FLAMINGO_info = median_and_scatter_model(selec_file, hmf_file)

names = [
    "5p6_based_counts_1e4.npy",
    "5p6_based_counts_3e5.npy",
    "5p6_based_counts_1e5.npy",
]

for k in range(3):
    FLAMINGO_model = ccm(FLAMINGO_info, SZ_cut[k], true_halo_mass_function=True)
    ns = []
    for i in tqdm(range(20)):
        ns.append(FLAMINGO_model.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0])
    ns = np.array(ns)
    np.save("data_to_fit_to/" + names[k], ns)


suffixes = ["1e4", "3e5", "1e5"]

for k in range(3):
    FLAMINGO_info_bhmf = median_and_scatter_model(
        selec_file, hmf_file, hydro_path_for_hmf_ratio=selec_file
    )
    FLAMINGO_model_bhmf = ccm(
        FLAMINGO_info_bhmf,
        SZ_cut[k],
        true_halo_mass_function=True,
        use_hydro_hmf_ratio=True,
    )
    ns = []
    for i in tqdm(range(20)):
        ns.append(FLAMINGO_model_bhmf.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0])
    ns = np.array(ns)
    np.save("data_to_fit_to/5p6_based_counts_baryon_hmf" + names[k], ns)

    FLAMINGO_info_bhmf = median_and_scatter_model(
        dict_folder + "hist_and_med_L1000N1800_JETS_Y.pkl",
        hmf_file,
        hydro_path_for_hmf_ratio=dict_folder + "hist_and_med_L1000N1800_JETS_Y.pkl",
    )
    FLAMINGO_model_bhmf = ccm(
        FLAMINGO_info_bhmf,
        SZ_cut[k],
        true_halo_mass_function=True,
        use_hydro_hmf_ratio=True,
    )
    ns = []
    for i in tqdm(range(20)):
        ns.append(FLAMINGO_model_bhmf.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0])
    ns = np.array(ns)
    np.save("data_to_fit_to/5p6_based_counts_jet_hmf" + names[k], ns)

    FLAMINGO_info_bhmf = median_and_scatter_model(
        dict_folder + "hist_and_med_L1000N1800_STRONGEST_Y.pkl",
        hmf_file,
        hydro_path_for_hmf_ratio=dict_folder
        + "hist_and_med_L1000N1800_STRONGEST_Y.pkl",
    )
    FLAMINGO_model_bhmf = ccm(
        FLAMINGO_info_bhmf,
        SZ_cut[k],
        true_halo_mass_function=True,
        use_hydro_hmf_ratio=True,
    )
    ns = []
    for i in tqdm(range(20)):
        ns.append(FLAMINGO_model_bhmf.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0])
    ns = np.array(ns)
    np.save("data_to_fit_to/5p6_based_counts_strongest_hmf" + names[k], ns)
