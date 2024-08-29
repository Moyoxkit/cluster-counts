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


survey = "SIO"
colors = ["black", "#117733", "#7EFF4B", "#105ba4"]

style.use("../pipeline-configs/colibre/mnras.mplstyle")
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fit_params = np.load("fit_params.npy")
best_fit = [0.306, 0.807, np.log10(fit_params[0]), fit_params[1], 0.081]
prior_errs = [0, 0, 0.02, 0.08, 0.01]
print(best_fit)

readr_fid = emcee.backends.HDFBackend("fit_outputs/" + survey + "_FID.hdf5")
chain_fid = readr_fid.get_chain(discard=500, flat=True)
ln_prob_fid = readr_fid.get_log_prob(discard=500, flat=True)
print(get_percent_away_from_truth([0.306, 0.807], chain_fid[:, :2], ln_prob_fid))

readr_bar = emcee.backends.HDFBackend("fit_outputs/" + survey + "_BAR.hdf5")
chain_bar = readr_bar.get_chain(discard=500, flat=True)
ln_prob_bar = readr_bar.get_log_prob(discard=500, flat=True)
print(get_percent_away_from_truth([0.306, 0.807], chain_bar[:, :2], ln_prob_bar))

readr_jet = emcee.backends.HDFBackend("fit_outputs/" + survey + "_JET.hdf5")
chain_jet = readr_jet.get_chain(discard=500, flat=True)
ln_prob_jet = readr_jet.get_log_prob(discard=500, flat=True)
print(get_percent_away_from_truth([0.306, 0.807], chain_jet[:, :2], ln_prob_jet))

readr_str = emcee.backends.HDFBackend("fit_outputs/" + survey + "_STR.hdf5")
chain_str = readr_str.get_chain(discard=500, flat=True)
ln_prob_str = readr_str.get_log_prob(discard=500, flat=True)
print(get_percent_away_from_truth([0.306, 0.807], chain_str[:, :2], ln_prob_str))


figure = plt.figure(figsize=(7, 7))
figure.set_tight_layout(True)
corner.corner(
    chain_fid,
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
    color=colors[0],
    plot_contours=True,
    plot_datapoints=False,
    plot_density=False,
    show_titles=False,
    fontsize=12,
    fig=figure,
)
corner.corner(
    chain_bar,
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
    color=colors[1],
    plot_contours=True,
    plot_datapoints=False,
    plot_density=False,
    show_titles=False,
    fontsize=12,
    fig=figure,
)
corner.corner(
    chain_jet,
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
    color=colors[2],
    plot_contours=True,
    plot_datapoints=False,
    plot_density=False,
    show_titles=False,
    fontsize=12,
    fig=figure,
)
corner.corner(
    chain_str,
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
    color=colors[3],
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
    if i > 1:
        x_vals = np.linspace(
            best_fit[i] - 5 * prior_errs[i], best_fit[i] + 5 * prior_errs[i], 100
        )
        ax.plot(
            x_vals,
            amps[i] * np.exp(-0.5 * (x_vals - best_fit[i]) ** 2 / (prior_errs[i] ** 2)),
            color="grey",
            ls="--",
            alpha=0.5,
        )
#     ax.axvline(s0[i], color="orange")

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(best_fit[xi], color="grey", ls="--", alpha=0.5)
        ax.axhline(best_fit[yi], color="grey", ls="--", alpha=0.5)

custom_legend = []
for i in range(4):
    custom_legend.append(Line2D([0], [0], color=colors[i]))

figure.legend(
    custom_legend,
    ["Fiducial", "Baryonic HMF", "Jet", r"fgas-8$\sigma$"],
    loc="center",
    bbox_to_anchor=(0.73, 0.75),
    fontsize=12,
)

plt.savefig("figures/cornerplot" + survey + ".pdf")
