import numpy as np
import corner
import matplotlib.pyplot as plt
import emcee
from matplotlib.pyplot import style
from matplotlib.lines import Line2D


def get_percent_away_from_truth(truth, samples, ln_prob):
    truth = np.array(truth)

    sample_med_0 = np.median(samples[:, 0])
    sample_med_1 = np.median(samples[:, 1])
    medians = np.array([sample_med_0, sample_med_1])

    sample_mean_0 = np.mean(samples[:, 0])
    sample_mean_1 = np.mean(samples[:, 1])
    medians = np.array([sample_mean_0, sample_mean_1])

    maxs = samples[np.argmax(ln_prob), :]

    percent_defs = np.abs(
        np.array(
            [
                (truth[0] - sample_med_0) / truth[0],
                (truth[1] - sample_med_1) / truth[1],
            ]
        )
    )
    cov_mat = np.cov(samples.T)
    distance = np.sqrt(
        np.matmul(np.matmul((truth - maxs).T, np.linalg.inv(cov_mat)), (truth - maxs))
    )

    return percent_defs, distance


colors = ["black", "#117733", "#7EFF4B", "#105ba4"]

style.use("../pipeline-configs/colibre/mnras.mplstyle")
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fit_params = np.load("fit_params.npy")
best_fit = [0.306, 0.807]
print(best_fit)

readr_spt = emcee.backends.HDFBackend("fit_outputs/SPT_HMF.hdf5")
chain_spt = readr_spt.get_chain(discard=500, flat=True)
ln_prob_spt = readr_spt.get_log_prob(discard=500, flat=True)
print(get_percent_away_from_truth([0.306, 0.807], chain_spt[:, :2], ln_prob_spt))

readr_sio = emcee.backends.HDFBackend("fit_outputs/SIO_HMF.hdf5")
chain_sio = readr_sio.get_chain(discard=500, flat=True)
ln_prob_sio = readr_sio.get_log_prob(discard=500, flat=True)
print(get_percent_away_from_truth([0.306, 0.807], chain_sio[:, :2], ln_prob_sio))


figure = plt.figure(figsize=(3.321, 3.321))
figure.set_tight_layout(True)
corner.corner(
    chain_spt,
    labels=[
        r"$\Omega_{\rm m}$",
        r"$\sigma_8$",
        r"log$_{10}~Y^*$",
        r"$\alpha$",
        r"log$_{10} \sigma$",
    ],
    bins=15,
    hist_bin_factor=2,
    levels=(0.68, 0.95),
    smooth1d=0,
    smooth=False,
    max_n_ticks=5,
    color=cycle[1],
    plot_contours=True,
    plot_datapoints=False,
    plot_density=False,
    show_titles=False,
    fontsize=12,
    fig=figure,
)
corner.corner(
    chain_sio,
    labels=[
        r"$\Omega_{\rm m}$",
        r"$\sigma_8$",
        r"log$_{10}~Y^*$",
        r"$\alpha$",
        r"log$_{10} \sigma$",
    ],
    bins=15,
    hist_bin_factor=2,
    levels=(0.68, 0.95),
    smooth1d=0,
    smooth=False,
    max_n_ticks=5,
    color=cycle[2],
    plot_contours=True,
    plot_datapoints=False,
    plot_density=False,
    show_titles=False,
    fontsize=12,
    fig=figure,
)

ndim = len(best_fit)
axes = np.array(figure.axes).reshape((ndim, ndim))
amps = [1, 1, 9000, 9000, 9000]

# Loop over the diagonal
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(best_fit[i], color="grey", ls="--", alpha=0.5)
#     ax.axvline(s0[i], color="orange")

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(best_fit[xi], color="grey", ls="--", alpha=0.5)
        ax.axhline(best_fit[yi], color="grey", ls="--", alpha=0.5)

custom_legend = []
for i in range(2):
    custom_legend.append(Line2D([0], [0], color=cycle[i + 1]))

figure.legend(
    custom_legend,
    ["SPT", "SO"],
    loc="center",
    bbox_to_anchor=(0.73, 0.75),
    fontsize=12,
)

plt.savefig("figures/cornerplot_HMF.pdf")
