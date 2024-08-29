import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from matplotlib.lines import Line2D

style.use("../colibre-stylesheer.mplsht")

baseline_counts_e4 = np.load("data_to_fit_to/5p6_based_counts_1e4.npy")
baseline_counts_3e5 = np.load("data_to_fit_to/5p6_based_counts_3e5.npy")
baseline_counts_e5 = np.load("data_to_fit_to/5p6_based_counts_1e5.npy")

baseline_counts_e4 = np.insert(baseline_counts_e4, 0, 0)[:21]
baseline_counts_3e5 = np.insert(baseline_counts_3e5, 0, 0)[:21]
baseline_counts_e5 = np.insert(baseline_counts_e5, 0, 0)[:21]

sky_factor = [1, 5200 / 41253, 0.4]
xmax = [0.9, 1.7, 2]

baseline_data = [
    baseline_counts_e4 * sky_factor[0],
    baseline_counts_3e5 * sky_factor[1],
    baseline_counts_e5 * sky_factor[2],
]

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
    all_axes[0].text(0.13, 0.43, "Planck", fontsize=10)
    all_axes[1].text(0.16, 0.43, "SPT", fontsize=10)
    all_axes[2].text(0.17, 0.43, "SO", fontsize=10)

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
plt.savefig("figures/Scaling_relations_percent.pdf")
