import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from matplotlib.lines import Line2D

style.use("../colibre-stylesheer.mplsht")

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
    r"M*$-\sigma\_$fgas$-4\sigma$",
    r"Jet",
    r"Jet$\_$fgas$-4\sigma$",
    "Planck",
    "PlanckNu0p24Fix ",
    "PlanckNu0p24Var",
    "LS8",
]
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

FLAMINGO_plots = {name: color for name, color in zip(FLAMINGO_labels, FLAMINGO_colors)}

FLAMINGO_colors = np.array(FLAMINGO_colors)[np.array([0, 4, 5, 6, 7, 10, 11, 12, 15])]
FLAMINGO_labels = np.array(FLAMINGO_labels)[np.array([0, 4, 5, 6, 7, 10, 11, 12, 15])]

pan_1_labels = ["Tinker (2010)", "Bocquet (2016)", "MiraTitanEmulator"]
pan_3_labels = ["LN", "PL", "LN+PL", "LN+PL+B16"]
pan_1_colors = [cycle[0], cycle[1], cycle[2]]
pan_3_colors = [cycle[3], cycle[4], cycle[5], cycle[6]]

all_labels = [pan_1_labels, FLAMINGO_labels, pan_3_labels, FLAMINGO_labels]
all_colors = [pan_1_colors, FLAMINGO_colors, pan_3_colors, FLAMINGO_colors]


linestyle = ["-", "--"]

redshift_edges = np.linspace(0, 3.0, 31)
baseline_counts_e4 = np.load("data_to_fit_to/5p6_based_counts_1e4.npy")
baseline_counts_3e5 = np.load("data_to_fit_to/5p6_based_counts_3e5.npy")
baseline_counts_e5 = np.load("data_to_fit_to/5p6_based_counts_1e5.npy")

baseline_counts_e4 = np.insert(baseline_counts_e4, 0, 0)[:21]
baseline_counts_3e5 = np.insert(baseline_counts_3e5, 0, 0)[:21]
baseline_counts_e5 = np.insert(baseline_counts_e5, 0, 0)[:21]

with open("data_for_crazy_figure_y.pkl", "rb") as f:
    data_for_fig_one = pickle.load(f)

print(data_for_fig_one)

plt.rcParams["font.size"] = 10

fig = plt.figure(figsize=(7, 7))

all_first_axes = []
all_second_axes = []
for i in range(4):
    all_first_axes.append(fig.add_subplot(2, 2, i + 1))
    all_second_axes.append(all_first_axes[i].twinx())

all_axes = [all_first_axes, all_second_axes]

z = np.linspace(0.0, 2.0, 21)

sky_factor = [5200 / 41253, 0.4]


print(np.sum(baseline_counts_3e5 * sky_factor[0]))
print(np.sum(baseline_counts_e5 * sky_factor[1]))

for cut_index in range(2):
    axes = all_axes[cut_index]
    data_for_cut = data_for_fig_one[cut_index + 1]
    for panel_index in range(len(data_for_cut)):
        ax = axes[panel_index]
        data_for_panel = data_for_cut[panel_index]
        for model_index in range(len(data_for_panel)):
            if panel_index == 3 and model_index == 0:
                continue
            ax.plot(
                z,
                np.insert(data_for_panel[model_index], 0, 0) * sky_factor[cut_index],
                ls=linestyle[cut_index],
                color=all_colors[panel_index][model_index],
                label=all_labels[panel_index][model_index],
            )

for ind, ax in enumerate(all_first_axes):
    ax.errorbar(
        z,
        baseline_counts_3e5 * sky_factor[0],
        np.sqrt(baseline_counts_3e5 * sky_factor[0]),
        color="black",
        label="Fiducial",
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 380)
    ax.set_xlim(0, 2)
    if ind > 1:
        ax.set_xlabel(r"$z$")
    else:
        ax.get_xaxis().set_visible(False)
    if ind == 1 or ind == 3:
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_ylabel(r"d$N_{\rm SPT}$ / d$z$")

for ind, ax in enumerate(all_second_axes):
    eb = ax.errorbar(
        z,
        baseline_counts_e5 * sky_factor[1],
        np.sqrt(baseline_counts_e5 * sky_factor[1]),
        color="black",
        label="Fiducial",
        ls="--",
    )
    ax.set_ylim(0, 3800)
    ax.set_xlim(0, 2)
    if ind == 0 or ind == 2:
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_ylabel(r"d$N_{\rm SO}$ / d$z$")

plt.setp(all_axes[0][0].get_yticklabels()[0], visible=False)
plt.setp(all_axes[1][1].get_yticklabels()[0], visible=False)

all_axes[0][1].spines["right"].set(ls=":", lw=1.5)
# all_axes[1][1].spines["right"].set(ls=":",lw=0)

all_axes[0][3].spines["right"].set(ls=":", lw=1.5)
# all_axes[1][3].spines["right"].set(ls=":",lw=0)

all_axes[0][1].legend(fontsize=11, title="Baryonic HMF modification")
all_axes[0][3].legend(fontsize=11, title="Scaling relation")

all_axes[0][3].set_xticks([0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

plt.subplots_adjust(left=0, right=1, bottom=0.0, top=1, wspace=0, hspace=0)
plt.close()
# plt.savefig("four_panel.png",dpi=300)

with open("data_for_crazy_figure_y.pkl", "rb") as f:
    data_for_fig_one = pickle.load(f)

plt.rcParams["font.size"] = 10

fig = plt.figure(figsize=(7, 8))

all_first_axes = []

all_planck_axes = [
    fig.add_subplot(4, 3, 1),
    fig.add_subplot(4, 3, 4),
    fig.add_subplot(4, 3, 7),
    fig.add_subplot(4, 3, 10),
]

all_SPT_axes = [
    fig.add_subplot(4, 3, 2),
    fig.add_subplot(4, 3, 5),
    fig.add_subplot(4, 3, 8),
    fig.add_subplot(4, 3, 11),
]

all_SO_axes = [
    fig.add_subplot(4, 3, 3),
    fig.add_subplot(4, 3, 6),
    fig.add_subplot(4, 3, 9),
    fig.add_subplot(4, 3, 12),
]

all_axes = [all_planck_axes, all_SPT_axes, all_SO_axes]

z = np.linspace(0.0, 2.0, 21)

sky_factor = [1, 5200 / 41253, 0.4]


print(np.sum(baseline_counts_3e5 * sky_factor[0]))
print(np.sum(baseline_counts_e5 * sky_factor[1]))

baseline_data = [
    baseline_counts_e4 * sky_factor[0],
    baseline_counts_3e5 * sky_factor[1],
    baseline_counts_e5 * sky_factor[2],
]

xmax = [0.9, 1.7, 2]

for i in range(4):
    all_axes[0][i].set_xticks([0.05, 0.25, 0.50, 0.75])
    all_axes[1][i].set_xticks([0.05, 0.50, 1.0, 1.5])
    all_axes[2][i].set_xticks([0.05, 0.50, 1.0, 1.5, 2.0])

for cut_index in range(3):
    data_for_cut = data_for_fig_one[cut_index]
    for panel_index in range(len(data_for_cut)):
        ax = all_axes[cut_index][panel_index]
        data_for_panel = data_for_cut[panel_index]
        for model_index in range(len(data_for_panel)):
            if panel_index == 3 and model_index == 0:
                continue
            if cut_index == 2:
                ax.plot(
                    z,
                    (
                        (
                            np.insert(data_for_panel[model_index], 0, 0)
                            * sky_factor[cut_index]
                            / baseline_data[cut_index]
                        )
                        - 1
                    ),  # * np.sqrt(baseline_data[cut_index]),
                    color=all_colors[panel_index][model_index],
                    label=all_labels[panel_index][model_index],
                )
            else:
                ax.plot(
                    z,
                    (
                        (
                            np.insert(data_for_panel[model_index], 0, 0)
                            * sky_factor[cut_index]
                            / baseline_data[cut_index]
                        )
                        - 1
                    ),  # * np.sqrt(baseline_data[cut_index]),
                    color=all_colors[panel_index][model_index],
                )
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlim(0.1, xmax[cut_index])
            if cut_index > 0:
                ax.get_yaxis().set_tick_params(labelleft=False)
            else:
                ax.set_ylabel(
                    r"(d$N$ / d$z$) / (d$N_{\rm FID}$ / d$z$) - 1", fontsize=10
                )
            if panel_index < 3:
                ax.get_xaxis().set_tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$z$", fontsize=10)
            if cut_index == 2:
                ax.legend(fontsize=8, loc=2)
            if model_index == 1:
                ax.fill_between(
                    z,
                    +1 / np.sqrt(baseline_data[cut_index]),
                    -1 / np.sqrt(baseline_data[cut_index]),
                    color="black",
                    alpha=0.05,
                )
                ax.axhline(0, color="black", ls=":")


all_axes[2][0].legend(fontsize=8, loc=1)
all_axes[2][1].legend(fontsize=8, ncol=2, loc=2)
all_axes[2][3].legend(fontsize=8, ncol=2, loc=2)

for i in range(1):
    all_axes[0][i].text(0.13, 0.4, "Planck", fontsize=10)
    all_axes[1][i].text(0.16, 0.4, "SPT", fontsize=10)
    all_axes[2][i].text(0.17, 0.4, "SO", fontsize=10)

plt.subplots_adjust(left=0, right=1, bottom=0.0, top=1, wspace=0, hspace=0)
plt.savefig("figures/four_panel_big_rat_fig.pdf")

with open("data_for_crazy_figure_y.pkl", "rb") as f:
    data_for_fig_one = pickle.load(f)

plt.rcParams["font.size"] = 10

all_axes = [all_first_axes, all_second_axes]

z = np.linspace(0.0, 2.0, 21)

sky_factor = [1, 5200 / 41253, 0.4]

linestyle = [":", "-", "--"]

print(np.sum(baseline_counts_e4 * sky_factor[0]))
print(np.sum(baseline_counts_e5 * sky_factor[1]))

fig_names = [
    "HMF_dn_Dz.pdf",
    "HMF_bar_dn_Dz.pdf",
    "scal_PL_dn_Dz.pdf",
    "scal_bar_dn_Dz.pdf",
]

for panel_index in range(4):
    fig = plt.figure(figsize=(4, 3.321))
    firsax = fig.add_subplot(111)
    twinax = firsax.twinx()
    both_axis = [firsax, firsax, twinax]
    for cut_index in range(3):
        data_for_panel = data_for_fig_one[cut_index][panel_index]
        ax = both_axis[cut_index]
        for model_index in range(len(data_for_panel)):
            if panel_index == 3 and model_index == 0:
                continue
            if cut_index == 0:
                label_to_use = None
            else:
                label_to_use = all_labels[panel_index][model_index]
            ax.plot(
                z,
                np.insert(data_for_panel[model_index], 0, 0) * sky_factor[cut_index],
                ls=linestyle[cut_index],
                color=all_colors[panel_index][model_index],
                label=label_to_use,
            )
    firsax.errorbar(
        z,
        baseline_counts_3e5 * sky_factor[1],
        np.sqrt(baseline_counts_3e5 * sky_factor[1]),
        color="black",
        label="Fiducial",
    )
    firsax.errorbar(
        z,
        baseline_counts_e4 * sky_factor[0],
        np.sqrt(baseline_counts_e4 * sky_factor[0]),
        color="black",
        ls=":",
    )
    firsax.set_ylim(0, 380)
    firsax.set_xlim(0, 2)
    firsax.set_xlabel(r"$z$")
    firsax.set_ylabel(r"d$N_{\rm Planck, SPT}$ / d$z$")

    twinax.errorbar(
        z,
        baseline_counts_e5 * sky_factor[2],
        np.sqrt(baseline_counts_e5 * sky_factor[2]),
        color="black",
        ls="--",
    )
    twinax.set_ylim(0, 3800)
    twinax.set_xlim(0, 2)
    twinax.set_xlabel(r"$z$")
    twinax.set_ylabel(r"d$N_{\rm SO}$ / d$z$")

    firsax.legend(fontsize=9)

    twinax.spines["right"].set(ls=":", lw=1.5)
    firsax.spines["right"].set(ls=":", lw=1.5)
    custom_lines = [
        Line2D([0], [0], color="black", ls=":"),
        Line2D([0], [0], color="black", ls="-"),
        Line2D([0], [0], color="black", ls="--"),
    ]
    twinax.legend(custom_lines, ["Planck", "SPT", "SO"], loc=2, fontsize=9)
    plt.savefig("figures/" + fig_names[panel_index])

fig = plt.figure(figsize=(3.321, 3.321))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

baseline_data = [baseline_counts_e4, baseline_counts_3e5, baseline_counts_e5]
base_labels = [r"PLANCK-like", "SPT-like", "SO-like"]

renormfacs = [1, 1, 1]
sky_factor = [1, 5200 / 41253, 0.4]
lines = [":", "-", "--"]
colors = [cycle[0], cycle[1], cycle[2]]
for_legend = []

for_legend = []

for ind, data in enumerate(baseline_data[:-1]):
    print(np.sum(data * sky_factor[ind]))
    ax.errorbar(
        z,
        data * sky_factor[ind] / renormfacs[ind],
        np.sqrt(data * sky_factor[ind]) / renormfacs[ind],
        label=base_labels[ind],
        ls=lines[ind],
        color=colors[ind],
    )
ax2.errorbar(
    z,
    baseline_data[2] * sky_factor[2] / renormfacs[2],
    np.sqrt(data * sky_factor[2]) / renormfacs[2],
    label=base_labels[2],
    ls=lines[2],
    color=colors[2],
)


ax.set_xlabel(r"$z$", fontsize=10)
ax.set_ylim(0, 330)
ax2.set_ylim(0, 3300)
ax.set_xlim(0, 2)

ax.set_ylabel(r"d$N_{\rm Planck, SPT}$ / d$z$", fontsize=10)
ax2.set_ylabel(r"d$N_{\rm SO}$ / d$z$", fontsize=10)
ax.spines["right"].set(ls=":", lw=1.5)
ax2.spines["right"].set(ls=":", lw=1.5)


custom_lines = [
    Line2D([0], [0], color=cycle[0], ls=":"),
    Line2D([0], [0], color=cycle[1], ls="-"),
    Line2D([0], [0], color=cycle[2], ls="--"),
]
ax.legend(custom_lines, base_labels, fontsize=9)
plt.savefig("figures/Diff_surveys.pdf")
