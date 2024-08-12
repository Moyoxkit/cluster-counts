import numpy as np
import pickle
import unyt
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model
from matplotlib.pyplot import style

dict_folder = "../make_hbt_pickles/"

style.use("../pipeline-configs/colibre/mnras.mplstyle")
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import pandas as pd

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
    "L1$\_$m9",
    "L2p8$\_$m9",
    "L1$\_$m10",
    "L1$\_$m8",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    r"M*$-\sigma$",
    r"M*$-\sigma+$fgas$-4\sigma$",
    "JETS",
    "JETS fgas$-4\sigma$",
    "Planck",
    "PlanckNu0p24Fix ",
    "PlanckNu0p24Var",
    "LS8",
]

FLAMINGO_plots = {name: color for name, color in zip(FLAMINGO_labels, FLAMINGO_colors)}

FLAMINGO_colors = np.array(FLAMINGO_colors)[np.array([0, 4, 5, 6, 7, 10, 11, 12, 15])]
FLAMINGO_labels = np.array(FLAMINGO_labels)[np.array([0, 4, 5, 6, 7, 10, 11, 12, 15])]

med_catalogues = np.array(
    [
        dict_folder + "hist_and_med_L1000N1800_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_WEAK_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONG_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONGER_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONGEST_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_JETS_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONG_JETS_Y.pkl",
    ]
)

fit_params = np.load("fit_params.npy")
scatter_param = 0.08190972605801156

baseline_info = median_and_scatter_model(
    dict_folder + "hist_and_med_L1000N1800_Y.pkl",
    dict_folder + "hist_and_med_L5600N5040DMO.pkl",
)

panel_two_infos = []
panel_for_infos = []
for file_loc in med_catalogues:
    panel_two_infos.append(
        median_and_scatter_model(
            dict_folder + "hist_and_med_L1000N1800_Y.pkl",
            DMO_HMF_dictionary_path=dict_folder + "hist_and_med_L5600N5040DMO.pkl",
            hydro_path_for_hmf_ratio=file_loc,
        )
    )
    panel_for_infos.append(
        median_and_scatter_model(
            file_loc,
            dict_folder + "hist_and_med_L5600N5040DMO.pkl",
        )
    )

all_cuts = []
n_z_bins = 20
d_z = 0.1
for SZ_cut in [1e-4, 3e-5, 1e-5]:
    panel_one_models = [
        ccm(baseline_info, SZ_cut, fit_model="Tinker10"),
        ccm(baseline_info, SZ_cut, fit_model="Bocquet500cDMOnly"),
        ccm(baseline_info, SZ_cut, mira_titan=True),
    ]
    panel_tre_models = [
        ccm(
            baseline_info,
            SZ_cut,
            power_law_meds=False,
            log_normal_scatter=True,
            true_halo_mass_function=True,
            power_law_args=fit_params,
            log_normal_lognsigy=scatter_param,
        ),
        ccm(
            baseline_info,
            SZ_cut,
            power_law_meds=True,
            log_normal_scatter=False,
            true_halo_mass_function=True,
            power_law_args=fit_params,
            log_normal_lognsigy=scatter_param,
        ),
        ccm(
            baseline_info,
            SZ_cut,
            power_law_meds=True,
            log_normal_scatter=True,
            true_halo_mass_function=True,
            power_law_args=fit_params,
            log_normal_lognsigy=scatter_param,
        ),
        ccm(
            baseline_info,
            SZ_cut,
            fit_model="Bocquet500cDMOnly",
            power_law_meds=True,
            log_normal_scatter=True,
            power_law_args=fit_params,
            log_normal_lognsigy=scatter_param,
        ),
    ]

    panel_two_models = []
    panel_for_models = []
    for model_index in range(len(med_catalogues)):
        panel_two_models.append(
            ccm(
                panel_two_infos[model_index],
                SZ_cut,
                true_halo_mass_function=True,
                use_hydro_hmf_ratio=True,
            )
        )
        panel_for_models.append(
            ccm(panel_for_infos[model_index], SZ_cut, true_halo_mass_function=True)
        )

    all_models = [
        panel_one_models,
        panel_two_models,
        panel_tre_models,
        panel_for_models,
    ]
    all_data = []
    for model_list in tqdm(all_models):
        all_panel_data = []
        for model in model_list:
            z = []
            counts_for_model = np.zeros(n_z_bins)
            for i in range(n_z_bins):
                z.append(i * 0.1)
                counts_for_model[i] = model.number_counts_sz(i * 0.1, (i + 1) * 0.1)[0]

            all_panel_data.append(counts_for_model)
        all_data.append(all_panel_data)
    all_cuts.append(all_data)

print(all_cuts)

with open("data_for_crazy_figure_y.pkl", "wb") as f:
    pickle.dump(all_cuts, f)
