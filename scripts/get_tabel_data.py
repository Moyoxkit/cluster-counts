import pickle
import warnings

warnings.filterwarnings("ignore")

with open("data_for_crazy_figure_y.pkl", "rb") as f:
    data_for_fig_one = pickle.load(f)

import numpy as np
from scipy.stats import norm, chi2

baseline_counts_e4 = np.load("data_to_fit_to/5p6_based_counts_1e4.npy")
baseline_counts_3e5 = np.load("data_to_fit_to/5p6_based_counts_3e5.npy")
baseline_counts_e5 = np.load("data_to_fit_to/5p6_based_counts_1e5.npy")

baseline_counts_e4 = np.insert(baseline_counts_e4, 0, 0)[:21]
baseline_counts_3e5 = np.insert(baseline_counts_3e5, 0, 0)[:21]
baseline_counts_e5 = np.insert(baseline_counts_e5, 0, 0)[:21]

FLAMINGO_labels = [
    r"L1$\_$m9",
    r"fgas$+2\sigma$",
    r"fgas$-2\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-8\sigma$",
    r"Jet",
    r"Jet$\_$fgas$-4\sigma$",
    "Planck",
    "PlanckNu0p24Fix ",
    "PlanckNu0p24Var",
    "LS8",
]

pan_1_labels = ["Tinker (2010)", "Bocquet (2016)", "MiraTitanEmulator"]
pan_3_labels = ["LN", "PL", "LN+PL", "LN+PL+MTE"]
all_labels = [pan_1_labels, FLAMINGO_labels, pan_3_labels, FLAMINGO_labels]


def convert_chi2_to_sigma(number_of_chi2, number_of_df):
    # Compute the p-value using the chi-squared CDF (1 - CDF since lower.tail=FALSE in R)
    p_value = chi2.sf(number_of_chi2, number_of_df)

    # Compute the log of the p-value
    log_p_value = np.log(p_value)

    # Convert the log-p-value back to a p-value
    p_value_from_log = np.exp(log_p_value)

    # Compute the number of sigma using the normal distribution quantile function
    number_of_sigma = norm.isf(p_value_from_log)

    return p_value


z = np.linspace(0.0, 2.0, 21)

sky_factor = [1, 5200 / 41253, 0.4]


baseline_data = [
    baseline_counts_e4 * sky_factor[0],
    baseline_counts_3e5 * sky_factor[1],
    baseline_counts_e5 * sky_factor[2],
]

xmax = [0.9, 1.7, 2.0]

# for k in range(3):
#     print(len(z[z < xmax[k]]) - 1)

for cut_index in range(3):
    data_for_cut = data_for_fig_one[cut_index]
    for panel_index in range(len(data_for_cut)):
        data_for_panel = data_for_cut[panel_index]
        for model_index in range(len(data_for_panel)):
            if panel_index == 3 and model_index == 0:
                continue
            else:
                data = np.insert(data_for_panel[model_index], 0, 0)
                data_mask = (z > 0) * (z < xmax[cut_index])
                chis2 = np.sum(
                    (
                        (data * sky_factor[cut_index] - baseline_data[cut_index]) ** 2
                        / (baseline_data[cut_index])
                    )[data_mask]
                )
                dof = [8, 16, 19][cut_index] - 1
                print(
                    "Survey = ",
                    cut_index,
                    " variation = ",
                    ["HMF", "HMF_bar", "scal_ass", "scal_var"][panel_index],
                    " model = ",
                    all_labels[panel_index][model_index],
                    " chi2 = ",
                    chis2,
                    " pval = ",
                    convert_chi2_to_sigma(chis2, dof),
                )
