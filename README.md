# cluster-counts
Code to a cluster count model to check against FLAMINGO

The code consists of a few classes that work together. For each python file, please read trough the code as it will give a detailed description of what parameters are available.

In the cluster model folder there are three files that are used for all the predictions

`median_and_scatter_model` handles the scaling relation, scatter and any values obtained directly from FLAMINGO. As an important input this uses a pickled dictionary that has the HMF, median and a histogram of the integrated compton Y which the code can use for predictions.

`cluster_model` handles the cosmology-dependent HMF models, and uses the data processed by median_and_scatter_model to actually predict number counts as a function of z.

`cosmology_fitter` can be used to MCMC fit Omega_m and sigma_8 using the cluster model. It includes priors, poissonian likelihoods and the MCMC itself.

The data folder has the data used for the paper, the different "Fiducial" cluster counts, the ones used for fitting and the counts for all the plots in the paper.

The scripts folder has all the scripts needed to both create the data and plots in the paper, notable ones are:

`make_model_data` this uses the cluster model to create and save the counts that are predicted

`get_distributions` and `make_plot` these scripts read the SOAP catalogs and produce the data format required for the cluster model.
