import numpy as np
from cosmology_fitter import cosmology_fitter as cf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-o", "--output", type=str)
parser.add_argument("-s", "--surveyid", type=int)
args = parser.parse_args()


redshift_edges = np.linspace(0, 2.0, 21)
sky_arr = [1, 5200 / 41253, 0.4]
limt = [1e-4, 3e-5, 1e-5]
initial_guess = np.array([0.306, 0.807, -0.19, 1.79, 0.075])
fit_test = cf(
    "../Selection_effects/non_log_normal/hist_and_med_L1000N1800_Y.pkl", args.input
)
params_mcmc, samples, lnprob = fit_test.model_fit_paralel_full_model(
    initial_guess,
    redshift_edges,
    nSteps=2500,
    nDiscard=500,
    nwalkers=40,
    args=[limt[args.surveyid], sky_arr[args.surveyid]],
    n_threads=128,
    outputname=args.output,
    restart=False,
)
