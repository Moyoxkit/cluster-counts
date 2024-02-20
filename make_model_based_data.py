import pandas as pd
import numpy as np
import unyt
from tqdm import tqdm
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model

redshift_edges = np.linspace(0,3.0,31)

nr_of_square_degs =  0.4#5200/41253 
SZ_cut = 1e-5

FLAMINGO_info = median_and_scatter_model("../Selection_effects/non_log_normal/hist_and_med_L1000N1800_Y.pkl","../Selection_effects/non_log_normal/hist_and_med_L5600N5040DMO.pkl",
                                          hydro_path_for_hmf_ratio="../Selection_effects/non_log_normal/hist_and_med_L1000N1800_STRONGEST_Y.pkl")
FLAMINGO_model = ccm(FLAMINGO_info,SZ_cut,true_halo_mass_function=True,use_hydro_hmf_ratio=True)

ns = []

for i in tqdm(range(30)):
    ns.append(FLAMINGO_model.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0])

ns = np.array(ns)
#print(np.sum(ns)*nr_of_square_degs)
#np.save("5p6_based_counts_3e5.npy",ns)
