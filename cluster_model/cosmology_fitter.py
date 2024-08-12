import emcee
import numpy as np
from scipy.special import gammaln
import corner
import unyt
import pandas as pd
from tqdm import tqdm
from cluster_model import cluster_cosmology_model as ccm
from median_and_scatter_model import median_and_scatter_model as masm
from multiprocessing import Pool


class cosmology_fitter:
    """
    Class that has all the functionality to fit cosmological parameters using MCMC
    """

    def __init__(self, file_loc, counts_file, bary_hmf_file=None):
        """
        Initialize the data and redshift bins to use for MCMC
        """

        self.cluster_file_loc = file_loc
        self.FLAMINGO_info = masm(file_loc, hydro_path_for_hmf_ratio=bary_hmf_file)
        cluster_model = ccm(self.FLAMINGO_info, 1e-4, true_halo_mass_function=True)
        self.base_cosmology = cluster_model.get_cosmology()
        self.count_data = np.load(counts_file)

    def ln_prior(self, params):
        # mean, sigma = params
        priors = [
            [0.25875397025, 0.33422387824],
            [0.7, 0.9],
        ]  # have a flat prior where parameters cant go outside the boundaries
        for i in range(len(params)):
            if params[i] < priors[i][0] or params[i] > priors[i][1]:
                return -np.inf
        return 0.0

    def ln_likelihood_paralel(
        self,
        params,
        SZ_cut,
        power_law_median,
        log_normal_scatter,
        mira_titan,
        fit_model=None,
        solid_angle=4 * np.pi,
    ):
        """
        Returns the log Poissonion likelihood using the cosmological model
        """

        if params[0] <= self.base_cosmology["Omega_b"]:
            return -np.inf

        if self.ln_prior(params) == -np.inf:
            return -np.inf

        cosmology_sample = self.base_cosmology | {
            "Omega_m": params[0],
            "sigma_8": params[1],
        }
        cluster_model = ccm(
            self.FLAMINGO_info,
            SZ_cut,
            cosmo_info=cosmology_sample,
            power_law_meds=power_law_median,
            log_normal_scatter=log_normal_scatter,
            mira_titan=mira_titan,
            fit_model=fit_model,
        )

        log_like = 0
        for index in range(len(self.z_edges) - 1):
            model_counts = cluster_model.number_counts_sz(
                self.z_edges[index],
                self.z_edges[index + 1],
            )[0] * (solid_angle / (4 * np.pi))
            if model_counts == 0 or self.count_data[index] == 0:
                continue
            log_like += (
                self.count_data[index]
                * np.log(model_counts)
                * (solid_angle / (4 * np.pi))
                - model_counts
                - gammaln(self.count_data[index] * (solid_angle / (4 * np.pi)))
            )

        if np.isnan(log_like):
            return -np.inf

        return log_like

    def model_fit_paralel(
        self,
        initial_guess,
        z_edges,
        nSteps=2000,
        nDiscard=500,
        nwalkers=100,
        args=[1e-4, False, False, False, None, 1.584],
        n_threads=10,
        outputname=None,
        restart=False,
    ):
        # Set up the data
        self.z_edges = z_edges
        # Set up the MCMC sampler
        ndim = len(initial_guess)  # number of parameters being fit
        # start each walker in a small ball around the initial guess
        # use a Gaussian distribution with width 1% of initial guess
        pos = [
            initial_guess + np.random.normal(loc=0, scale=np.abs(initial_guess * 0.02))
            for j in range(nwalkers)
        ]

        # Set up support for restarts
        backend = emcee.backends.HDFBackend(outputname)
        if not restart:
            backend.reset(nwalkers, ndim)

        # initialize the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.ln_likelihood_paralel,
            args=args,
            pool=Pool(n_threads),
            backend=backend,
        )
        # run the sampler for 'nSteps' steps; results in "nWalkers * nSteps" samples
        sampler.run_mcmc(pos, nSteps, progress=True)

        # The output, sampler.chain has shape [nwalkers, nsteps, ndim]
        # the MCMC samples after discrading the first nDiscard ones
        samples = sampler.chain[:, nDiscard:, :].reshape((-1, ndim))
        # the model likelihood for each MCMC sample
        lnprob = sampler.lnprobability[:, nDiscard:].reshape(-1)

        # calculate the median and -1\sigma, +1\sigma confidence interval for each parameter
        params_mcmc = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84], axis=0)),
        )

        return params_mcmc, samples, lnprob

    def ln_prior_full_model(self, params):
        # mean, sigma = params
        priors = [
            [0.25875397025, 0.33422387824],
            [0.7, 0.9],
            [-5, 5],
            [0.01, 8.59],
            [0.005, 0.50],
        ]  # have a flat prior where parameters cant go outside the boundaries
        for i in range(len(params)):
            if params[i] < priors[i][0] or params[i] > priors[i][1]:
                return -np.inf
        return (
            -((-0.09839464277 - params[2]) ** 2) / ((0.02) ** 2)
            - ((1.65965115 - params[3]) ** 2) / ((0.08) ** 2)
            - ((0.08190972605801156 - params[4]) ** 2) / ((0.01) ** 2)
        )

    def ln_likelihood_paralel_full_model(self, params, SZ_cut, solid_angle):
        """
        Returns the log Poissonion likelihood using the cosmological model
        """

        if params[0] <= self.base_cosmology["Omega_b"]:
            return -np.inf

        if self.ln_prior_full_model(params) == -np.inf:
            return -np.inf

        cosmology_sample = self.base_cosmology | {
            "Omega_m": params[0],
            "sigma_8": params[1],
        }
        cluster_model = ccm(
            self.FLAMINGO_info,
            SZ_cut,
            cosmo_info=cosmology_sample,
            power_law_meds=True,
            log_normal_scatter=True,
            mira_titan=False,
            fit_model="Bocquet500cDMOnly",
            log_normal_lognsigy=params[4],
            power_law_args=(10 ** params[2], params[3], 0.88858989),
        )

        log_like = 0

        for index in range(len(self.z_edges) - 1):
            model_counts = cluster_model.number_counts_sz(
                self.z_edges[index],
                self.z_edges[index + 1],
            )[0] * (solid_angle / (4 * np.pi))
            if model_counts == 0 or self.count_data[index] == 0:
                continue
            log_like += (
                self.count_data[index]
                * np.log(model_counts)
                * (solid_angle / (4 * np.pi))
                - model_counts
                - gammaln(self.count_data[index] * (solid_angle / (4 * np.pi)))
            )

        if np.isnan(log_like):
            return -np.inf

        return log_like + self.ln_prior_full_model(params)

    def model_fit_paralel_full_model(
        self,
        initial_guess,
        z_edges,
        nSteps=2000,
        nDiscard=500,
        nwalkers=100,
        args=[1e-4, 4 * np.pi],
        n_threads=10,
        outputname=None,
        restart=False,
    ):
        # Set up the data
        self.z_edges = z_edges
        # Set up the MCMC sampler
        ndim = len(initial_guess)  # number of parameters being fit
        # start each walker in a small ball around the initial guess
        # use a Gaussian distribution with width 1% of initial guess
        pos = [
            initial_guess + np.random.normal(loc=0, scale=np.abs(initial_guess * 0.01))
            for j in range(nwalkers)
        ]

        # Set up support for restarts
        backend = emcee.backends.HDFBackend(outputname)
        if not restart:
            backend.reset(nwalkers, ndim)

        # initialize the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.ln_likelihood_paralel_full_model,
            args=args,
            pool=Pool(n_threads),
            backend=backend,
        )
        # run the sampler for 'nSteps' steps; results in "nWalkers * nSteps" samples
        sampler.run_mcmc(pos, nSteps, progress=True)

        # The output, sampler.chain has shape [nwalkers, nsteps, ndim]
        # the MCMC samples after discrading the first nDiscard ones
        samples = sampler.chain[:, nDiscard:, :].reshape((-1, ndim))
        # the model likelihood for each MCMC sample
        lnprob = sampler.lnprobability[:, nDiscard:].reshape(-1)

        # calculate the median and -1\sigma, +1\sigma confidence interval for each parameter
        params_mcmc = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84], axis=0)),
        )

        return params_mcmc, samples, lnprob
