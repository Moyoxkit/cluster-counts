import numpy as np
import unyt
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model
from matplotlib.pyplot import style

style.use("../pipeline-configs/colibre/mnras.mplstyle")
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import pandas as pd

percent_error = 0.1
all_cuts = []
n_z_bins = 20
d_z = 0.1

both_percents = []

selec_file = "../make_hbt_pickles/hist_and_med_L1000N1800_Y.pkl"
dmo_file = "../make_hbt_pickles/hist_and_med_L5600N5040DMO.pkl"
fit_params = np.load("fit_params.npy")
scatter_param = 0.08190972605801156

for prefactor in tqdm([-1, -0.5, -0.1, 0, 0.1, 0.5, 1]):
    all_cuts = []
    baseline_info_ys = median_and_scatter_model(
        selec_file,
        dmo_file,
    )
    baseline_info_ap = median_and_scatter_model(
        selec_file,
        dmo_file,
    )
    baseline_info_bt = median_and_scatter_model(
        selec_file,
        dmo_file,
    )
    for SZ_cut in [1e-4, 3e-5, 1e-5]:
        all_models = [
            ccm(
                baseline_info_ys,
                SZ_cut,
                power_law_meds=True,
                log_normal_scatter=True,
                true_halo_mass_function=True,
                power_law_args=[
                    10 ** (np.log10(fit_params[0]) * (1 + prefactor * percent_error)),
                    fit_params[1],
                    fit_params[2],
                ],
                log_normal_lognsigy=scatter_param,
            ),
            ccm(
                baseline_info_ap,
                SZ_cut,
                power_law_meds=True,
                log_normal_scatter=True,
                true_halo_mass_function=True,
                power_law_args=[
                    10 ** (np.log10(fit_params[0])),
                    fit_params[1] * (1 + prefactor * percent_error),
                    fit_params[2],
                ],
                log_normal_lognsigy=scatter_param,
            ),
            ccm(
                baseline_info_bt,
                SZ_cut,
                power_law_meds=True,
                log_normal_scatter=True,
                true_halo_mass_function=True,
                power_law_args=[
                    10 ** (np.log10(fit_params[0])),
                    fit_params[1],
                    fit_params[2] * (1 + prefactor * percent_error),
                ],
                log_normal_lognsigy=scatter_param,
            ),
        ]
        all_data = []
        for model_index in range(3):
            z = []
            counts_for_model = np.zeros(n_z_bins)
            for i in range(n_z_bins):
                z.append(i * 0.1)
                counts_for_model[i] = all_models[model_index].number_counts_sz(
                    i * 0.1, (i + 1) * 0.1
                )[0]
            all_data.append(counts_for_model)
        all_cuts.append(all_data)
    both_percents.append(all_cuts)

import pickle

f = open("data_for_percent_figure_y.pkl", "wb")
pickle.dump(both_percents, f)
f.close()

with open("data_for_percent_figure_y.pkl", "rb") as f:
    data_for_another_fig = pickle.load(f)

plt.rcParams["font.size"] = 10

fig = plt.figure(figsize=(7, 3.321))

all_axes = [
    fig.add_subplot(1, 3, 1),
    fig.add_subplot(1, 3, 2),
    fig.add_subplot(1, 3, 3),
]

colors = ["red", "black", "blue"]
linestyles = [":", "-", "--"]

z = np.linspace(0.0, 2.0, 21)

sky_factor = [1, 5200 / 41253, 0.4]

indexes_to_plot = [0, 3, -1]

for cut_index in range(3):
    for deviations_index in range(3):
        deviation_index = indexes_to_plot[deviations_index]
        for param_to_change_index in range(3):
            if deviations_index == 1:
                continue
            data_to_plot = np.insert(
                data_for_another_fig[deviation_index][cut_index][param_to_change_index],
                0,
                0,
            ) / np.insert(
                data_for_another_fig[indexes_to_plot[1]][cut_index][
                    param_to_change_index
                ],
                0,
                0,
            )
            all_axes[cut_index].plot(
                z,
                data_to_plot - 1,
                color=colors[deviations_index],
                linestyle=linestyles[param_to_change_index],
            )
            if deviations_index == 0:
                all_axes[cut_index].fill_between(
                    z,
                    +1 / np.sqrt(baseline_data[cut_index]),
                    -1 / np.sqrt(baseline_data[cut_index]),
                    color="black",
                    alpha=0.05,
                )
                all_axes[cut_index].axhline(0, color="black", ls="-.")
                all_axes[cut_index].set_xlabel(r"$z$")
            all_axes[cut_index].set_ylim(-0.5, 0.5)
            all_axes[cut_index].set_xlim(0.1, xmax[cut_index])
            if cut_index > 0:
                all_axes[cut_index].get_yaxis().set_tick_params(labelleft=False)
            if cut_index == 0:
                all_axes[cut_index].set_ylabel(
                    r"(d$N$ / d$z$) / (d$N_{\rm FID}$ / d$z$) - 1", fontsize=10
                )


for i in range(1):
    all_axes[0].text(0.13, 0.4, "Planck", fontsize=10)
    all_axes[1].text(0.16, 0.4, "SPT", fontsize=10)
    all_axes[2].text(0.17, 0.4, "SO", fontsize=10)

custom_lines = [
    Line2D([0], [0], color="black", ls=":"),
    Line2D([0], [0], color="black", ls="-"),
    Line2D([0], [0], color="black", ls="--"),
]
all_axes[0].legend(
    custom_lines, [r"$Y_{*}$", r"$\alpha$", r"$\beta$"], loc=3, fontsize=9
)

custom_lines = [
    Line2D([0], [0], color="red", ls="-"),
    Line2D([0], [0], color="blue", ls="-"),
]

all_axes[1].legend(custom_lines, [r"+$10\%$", r"-10$\%$"], loc=3, fontsize=9)
plt.savefig("Scaling_relations_percent.pdf")
