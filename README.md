# cluster-counts
Code to a simple cluster count model to check against FLAMINGO

The code consists of a few classes that work together. For each python file, please read trough the code as it will give a detailed description of what parameters are available.

`median_and_scatter_model` handles the scaling relation, scatter and any values obtained directly from FLAMINGO. As an important input this uses a pickled dictionary that has the HMF, median and a histogram of the integrated compton Y which the code can use for predictions.
`cluster_model` handles the cosmology-dependent HMF models, and uses the data processed by median_and_scatter_model to actually predict number counts as a function of z.
`cosmology_fitter` can be used to MCMC fit Omega_m and sigma_8 using the cluster model. It includes priors, poissonian likelihoods and the MCMC itself.
`make_model_based_data` uses the cluster model to quickly give you the redshift distribution of clusters at the FLAMINGO cosmology for any setup of the cluster model. Best used as an example.
`run_mcmc_chain` can be used as an example for how to run the full cosmology fitting. Read the arguments to see all the available options for fitting.
