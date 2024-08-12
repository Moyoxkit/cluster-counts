import pandas as pd
import numpy as np
import unyt
from tqdm import tqdm
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model

redshift_edges = np.linspace(0, 3.0, 31)

SZ_cut = [1e-4, 3e-5, 1e-5]

dict_folder = "../Selection_effects/non_log_normal/"

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
    for i in tqdm(range(30)):
        ns.append(FLAMINGO_model.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0])
    ns = np.array(ns)
    np.save("data_to_fit_to/" + names[k], ns)
