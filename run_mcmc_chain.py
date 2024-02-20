import numpy as np
from cosmology_fitter import cosmology_fitter as cf
import argparse

prefix = "/cosma8/data/dp004/dc-kuge1/cluster_cosmology/data_to_fit_to/"
catalogs = [prefix + "5p6_based_counts_1e4.npy",prefix + "5p6_based_counts_3e5.npy", prefix + "5p6_based_counts_1e5.npy"]
skycov = np.array([1,5200 / 41253,0.4])*4*np.pi
ycuts = [1e-4,3e-5,1e-5]

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--variation", type=str)
parser.add_argument("-s", "--survey", type=int)
parser.add_argument("-f", "--hmf",type=str)
parser.add_argument("-o", "--output",type=str)
parser.add_argument("-p", "--powerlaw",action='store_true')
parser.add_argument("-l", "--lognormal",action='store_true')
parser.add_argument("-m", "--miratitan",action='store_true')
args = parser.parse_args()

redshift_edges = np.linspace(0, 3.0, 31)

initial_guess = np.array([0.306, 0.807])
fit_test = cf(
    "/cosma8/data/dp004/dc-kuge1/Selection_effects/non_log_normal/hist_and_med_L1000N1800"
    + str(args.variation)
    + "Y.pkl",
    str(args.hmf),
    catalogs[int(args.survey)],
)

params_mcmc, samples, lnprob = fit_test.model_fit_paralel(
    initial_guess,
    redshift_edges,
    nSteps=5000,
    nDiscard=250,
    nwalkers=40,
    args=[ycuts[int(args.survey)], args.powerlaw, args.lognormal, args.miratitan, str(args.hmf), skycov[int(args.survey)]],
    n_threads=128,
    outputname=str(args.output),
    restart=False,
)
