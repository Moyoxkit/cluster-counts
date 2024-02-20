import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import dblquad, quad
from scipy.interpolate import (
    interp1d,
    LinearNDInterpolator,
    NearestNDInterpolator,
    griddata,
)
from scipy.special import erf
from hmf import MassFunction
import pandas as pd
import unyt
import pickle


class median_and_scatter_model:
    """
    This class generates the functions based on FLAMINGO data that never change
    when varying the cosmology
    """

    def __init__(
        self,
        dictionary_path,
        DMO_HMF_dictionary_path=None,
        PLLN_params=[10 ** -0.19, 1.79, 0.66, 0.075],
        hydro_path_for_hmf_ratio=None,
    ):
        """
        Initialises and sets up the interpolators
        for the FLAMINGO medians and scatter
        """

        self.dictionary_path = dictionary_path
        self.median_maker_flamingo(dictionary_path)
        self.scatter_maker_flamingo(dictionary_path)
        if DMO_HMF_dictionary_path == None:
            self.make_flamingo_halo_mass_function(dictionary_path)
        else:
            self.make_flamingo_halo_mass_function(DMO_HMF_dictionary_path)
        if hydro_path_for_hmf_ratio != None:
            self.make_flamingo_hmf_rat(hydro_path_for_hmf_ratio)
        self.Ystar = PLLN_params[0]
        self.alpha = PLLN_params[1]
        self.beta = PLLN_params[2]
        self.lognsigy = PLLN_params[3]

    def init_other_interpolators(self, astropy_cosmology):
        self.astropy_cosmology = astropy_cosmology
        self.scatter_maker_flamingo_LN(self.dictionary_path)
        self.scatter_maker_flamingo_PL(self.dictionary_path)
        self.scatter_maker_flamingo_LN_PL(self.dictionary_path)

    def median_maker_flamingo(self, dictionary_path):
        """
        Function that initialises and interpolates the FLAMINGO medians
        takes as an input the path to a dictionary that contains the FLAMINGO
        medians and scatter at each redshift and mass bin. The function
        unpacks this information and creates an interpolator in halo mass
        and redshift. Both halo mass and the median Compton-Y are fed interpolated
        in log space to increase accuracy.

        Parameters:

        dictionary_path : Path to the dictionary obtained from one of the FLAMINGO simulations
        """
        with open(dictionary_path, "rb") as f:
            all_medians = pickle.load(f)

        # Start by initialising the array that will be fed into the interpolator
        # All arrays are the same size as the interpolator takes as an input
        # the coordinate vectors of each component, even for a regular grid
        values_to_interpolate = np.zeros(
            (len(all_medians.keys()), len(all_medians["0"]) - 1)
        )
        redshifts = np.zeros((len(all_medians.keys()), len(all_medians["0"]) - 1))
        masses = np.zeros((len(all_medians.keys()), len(all_medians["0"]) - 1))

        # Loop over all redshift and mass bins and fill up the arrays
        for redshift_index in range(values_to_interpolate.shape[0]):
            current_median = all_medians[str(redshift_index)]
            redshifts[redshift_index, :] = current_median["z"]
            for mass_index in range(values_to_interpolate.shape[1] - 1):
                masses[:, mass_index] = current_median[str(mass_index)]["Mass_cut"]
                values_to_interpolate[redshift_index, mass_index] = current_median[
                    str(mass_index)
                ]["median"]

        # Move the relevant quantities to log space and filter out values
        # that lead to NaNs or negative infinities.
        masses = np.log10(masses)
        values_to_interpolate = np.log10(values_to_interpolate)
        values_to_interpolate[values_to_interpolate == -np.inf] = np.nan
        # Apply the filter to each array and flatten to be in the correct format for the interpolator
        values_to_interpolate = values_to_interpolate.flatten()
        masses = masses.flatten()[~np.isnan(values_to_interpolate)]
        redshifts = redshifts.flatten()[~np.isnan(values_to_interpolate)]
        values_to_interpolate = values_to_interpolate[~np.isnan(values_to_interpolate)]
        # Return -inf outside the interpolation range. Halos in those ranges don't exist
        # in the sim and setting the expected cy to zero is thus in line with the sims
        interpolator = LinearNDInterpolator(
            np.array([redshifts, masses]).T, values_to_interpolate, fill_value=-np.inf
        )

        self.flamingo_median_interpolator = interpolator

    def get_median_flamingo(self, M500, z):
        """
        Return the value of the flamingo medians.
        Required as the interpolation is done in log space
        """
        return 10 ** self.flamingo_median_interpolator(z, np.log10(M500))

    def scatter_maker_flamingo(self, dict_path):
        """
        Function that initialises and interpolates the fraction of clusters as
        a function of a cy cut for each input redshift and mass bin. As we use
        the input bins as out points to sample for the integration, we create
        a 2D array with an interpolar containing the fraction of objects above
        an input cy in each bin.

        Parameters:

        dictionary_path : Path to the dictionary obtained from one of the FLAMINGO simulations
        """
        with open(dict_path, "rb") as f:
            all_scatter = pickle.load(f)

        # Create an array with all the information needed for interpolation
        values_to_interpolate = np.zeros(
            (
                len(all_scatter.keys()),
                len(all_scatter["0"]) - 1,
                len(all_scatter["0"]["0"]["bins"]),
            )
        )
        # For the scatter we need only the coordinate axis which we save for later
        redshifts = np.zeros(values_to_interpolate.shape[0])
        masses = np.zeros(values_to_interpolate.shape[1])
        # We create a list that will hold all the interpolators
        interpolator_array = []
        for redshift_index in range(values_to_interpolate.shape[0]):
            current_scatter = all_scatter[str(redshift_index)]
            redshifts[redshift_index] = current_scatter["z"]
            # Create a temporary list that we later append to the interpolator list
            # thus creating a 2D list of interpolators for each z and mass
            temp_list = []
            for mass_index in range(values_to_interpolate.shape[1]):
                masses[mass_index] = current_scatter[str(mass_index)]["Mass_cut"]
                c_ys = current_scatter[str(mass_index)]["bins"]
                counts = current_scatter[str(mass_index)]["hist"]
                summed_counts = np.sum(counts)
                # Check if there are objects in the bin to avoid divide by zero
                # If there are no objects, the fraction should return zero
                if summed_counts == 0:
                    values_to_interpolate[redshift_index, mass_index, :] = 0
                else:
                    # This loop does an inverse cumulative sum
                    for bin_index in range(len(counts)):
                        values_to_interpolate[redshift_index, mass_index, bin_index] = (
                            np.sum(counts[bin_index:]) / summed_counts
                        )

                temp_list.append(
                    interp1d(
                        np.log10(current_scatter[str(mass_index)]["bins"]),
                        values_to_interpolate[redshift_index, mass_index, :],
                        fill_value=1,
                        bounds_error=False,
                    )
                )
            interpolator_array.append(temp_list)

        self.scatter_interpolators = interpolator_array
        self.all_redshifts = redshifts
        self.all_masses = masses
        self.c_ys = c_ys

    def make_flamingo_halo_mass_function(self, dictionary_path):
        """
        Creates an array that contains all the halo mass function points across mass
        and redshift.

        Parameters:
        dictionary_path : path to the pickle that has the hmd information
        """

        with open(dictionary_path, "rb") as f:
            all_medians = pickle.load(f)

        flamingo_number_densities = np.zeros(
            (len(all_medians.keys()), len(all_medians["0"]) - 1)
        )

        for redshift_index in range(flamingo_number_densities.shape[0]):
            current_median = all_medians[str(redshift_index)]
            for mass_index in range(flamingo_number_densities.shape[1] - 1):
                flamingo_number_densities[redshift_index, mass_index] = current_median[
                    str(mass_index)
                ]["dn_dlog10m"]

        self.flamingo_number_densities = flamingo_number_densities

    def power_law_sz_scaling_relation(self, M500, z, Ystar, alpha, beta):
        """
        Returns the value of the expected median Compton-Y given the
        parameters of a power law. Cosmology and z dependence taken
        from Planck

        Parameters:

        M500 : halo mass to evalute the relation at in Msun
        z    : redshift
        Ystar: normalisation of the power law
        alpha: slope of the power law
        beta : slope of the cosmology dependence
        """

        return (
            (self.astropy_cosmology.H(z) / self.astropy_cosmology.H0) ** (beta)
            * (self.astropy_cosmology.H(z) / (70 * u.km / (u.s * u.Mpc))) ** (alpha - 2)
            * (0.688 * M500 / 6e14) ** alpha
            * 1e-4
        )

    def halo_frac_sz_lognormal(
        self, lognsigy, M500, z, Ystar, alpha, beta, y_cut, FLAMINGO=False
    ):
        """
        Returns the fraction of halos in a mass bin defined by M500 that are
        part of a sample with cy cut y_cut assuming a log-normal distribution

        Parameters:

        M500     : halo mass to evaluate the relation in Msun
        z        : redshift
        Ystar    : normalisation of the power law (Used when FLAMINGO=False)
        alpha    : slope of mass-SZ relation (Used when FLAMINGO=False)
        beta     : slope of the redshift dependence (Used when FLAMINGO=False)
        y_cut    : the Compton Y cut that defines the sample
        FLAMINGO : use the FLAMINGO medians instead of the power law model
        """
        if FLAMINGO:
            return 0.5 * (
                1
                - erf(
                    (np.log10(y_cut) - np.log10(self.get_median_flamingo(M500, z)))
                    / (2 ** (0.5) * lognsigy)
                )
            )
        else:
            return 0.5 * (
                1
                - erf(
                    (
                        np.log10(y_cut)
                        - np.log10(
                            self.power_law_sz_scaling_relation(
                                M500, z, Ystar, alpha, beta
                            )
                        )
                    )
                    / (2 ** (0.5) * lognsigy)
                )
            )

    def scatter_maker_flamingo_LN(self, dict_path):
        """
        Function that initialises and interpolates the fraction of clusters as
        a function of a cy cut for each input redshift and mass bin. As we use
        the input bins as out points to sample for the integration, we create
        a 2D array with an interpolar containing the fraction of objects above
        an input cy in each bin. This uses log-normal scatter with the FLAMINGO
        medians

        Parameters:

        dictionary_path : Path to the dictionary obtained from one of the FLAMINGO simulations
        """
        # Create an array with all the information needed for interpolation
        values_to_interpolate = np.zeros(
            (
                len(self.all_redshifts),
                len(self.all_masses),
                len(self.c_ys),
            )
        )
        # We create a list that will hold all the interpolators
        interpolator_array = []
        for redshift_index in range(values_to_interpolate.shape[0]):
            # Create a temporary list that we later append to the interpolator list
            # thus creating a 2D list of interpolators for each z and mass
            temp_list = []
            for mass_index in range(values_to_interpolate.shape[1]):
                values_to_interpolate[
                    redshift_index, mass_index, :
                ] = self.halo_frac_sz_lognormal(
                    self.lognsigy,
                    self.all_masses[mass_index],
                    self.all_redshifts[redshift_index],
                    self.Ystar,
                    self.alpha,
                    self.beta,
                    self.c_ys,
                    FLAMINGO=True,
                )
                temp_list.append(
                    interp1d(
                        np.log10(self.c_ys),
                        values_to_interpolate[redshift_index, mass_index, :],
                        fill_value=1,
                        bounds_error=False,
                    )
                )
            interpolator_array.append(temp_list)

        self.scatter_interpolators_LN = interpolator_array

    def scatter_maker_flamingo_LN_PL(self, dict_path):
        """
        Function that initialises and interpolates the fraction of clusters as
        a function of a cy cut for each input redshift and mass bin. As we use
        the input bins as out points to sample for the integration, we create
        a 2D array with an interpolar containing the fraction of objects above
        an input cy in each bin. This uses log-normal scatter with power law
        medians

        Parameters:

        dictionary_path : Path to the dictionary obtained from one of the FLAMINGO simulations
        """
        # Create an array with all the information needed for interpolation
        values_to_interpolate = np.zeros(
            (
                len(self.all_redshifts),
                len(self.all_masses),
                len(self.c_ys),
            )
        )
        # We create a list that will hold all the interpolators
        interpolator_array = []
        for redshift_index in range(values_to_interpolate.shape[0]):
            # Create a temporary list that we later append to the interpolator list
            # thus creating a 2D list of interpolators for each z and mass
            temp_list = []
            for mass_index in range(values_to_interpolate.shape[1]):
                values_to_interpolate[
                    redshift_index, mass_index, :
                ] = self.halo_frac_sz_lognormal(
                    self.lognsigy,
                    self.all_masses[mass_index],
                    self.all_redshifts[redshift_index],
                    self.Ystar,
                    self.alpha,
                    self.beta,
                    self.c_ys,
                    FLAMINGO=False,
                )
                temp_list.append(
                    interp1d(
                        np.log10(self.c_ys),
                        values_to_interpolate[redshift_index, mass_index, :],
                        fill_value=1,
                        bounds_error=False,
                    )
                )
            interpolator_array.append(temp_list)

        self.scatter_interpolators_LN_PL = interpolator_array

    def scatter_maker_flamingo_PL(self, dict_path):
        """
        Function that initialises and interpolates the fraction of clusters as
        a function of a cy cut for each input redshift and mass bin. As we use
        the input bins as out points to sample for the integration, we create
        a 2D array with an interpolar containing the fraction of objects above
        an input cy in each bin. This uses Power-law with FLAMINGO scatter

        Parameters:

        dictionary_path : Path to the dictionary obtained from one of the FLAMINGO simulations
        """
        with open(dict_path, "rb") as f:
            all_scatter = pickle.load(f)

        # Create an array with all the information needed for interpolation
        values_to_interpolate = np.zeros(
            (
                len(self.all_redshifts),
                len(self.all_masses),
                len(self.c_ys),
            )
        )
        interpolator_array = []
        for redshift_index in range(values_to_interpolate.shape[0]):
            # Create a temporary list that we later append to the interpolator list
            # thus creating a 2D list of interpolators for each z and mass
            current_scatter = all_scatter[str(redshift_index)]
            temp_list = []
            for mass_index in range(values_to_interpolate.shape[1]):
                flamingo_median = self.get_median_flamingo(
                    self.all_masses[mass_index],
                    self.all_redshifts[redshift_index],
                )
                power_law_median = self.power_law_sz_scaling_relation(
                    self.all_masses[mass_index],
                    self.all_redshifts[redshift_index],
                    self.Ystar,
                    self.alpha,
                    self.beta,
                )
                c_y_shift = -(power_law_median - flamingo_median)
                c_ys = np.array(current_scatter[str(mass_index)]["bins"]) - c_y_shift
                counts = current_scatter[str(mass_index)]["hist"]
                summed_counts = np.sum(counts)
                # Check if there are objects in the bin to avoid divide by zero
                # If there are no objects, the fraction should return zero
                if summed_counts == 0:
                    values_to_interpolate[redshift_index, mass_index, :] = 0
                else:
                    # This loop does an inverse cumulative sum
                    for bin_index in range(len(counts)):
                        values_to_interpolate[redshift_index, mass_index, bin_index] = (
                            np.sum(counts[bin_index:]) / summed_counts
                        )
                temp_list.append(
                    interp1d(
                        np.log10(c_ys),
                        values_to_interpolate[redshift_index, mass_index, :],
                        fill_value=1,
                        bounds_error=False,
                    )
                )
            interpolator_array.append(temp_list)

        self.scatter_interpolators_PL = interpolator_array

    def make_flamingo_hmf_rat(self,hydro_file_path):
        """
        Define an array that has for every mass and z the ratio with the DMO HMF
        default use the 1Gpc DMO HMF as the interest is the FB vars
        """
        with open(
            "/cosma8/data/dp004/dc-kuge1/Selection_effects/non_log_normal/hist_and_med_L1000N1800DMO.pkl",
            "rb",
        ) as f:
            dmo_dictionary_with_hmf = pickle.load(f)
        with open(hydro_file_path, "rb") as f:
            hyd_dictionary_with_hmf = pickle.load(f)

        dmo_hmfs = np.zeros((len(self.all_redshifts), len(self.all_masses)))
        hyd_hmfs = np.zeros((len(self.all_redshifts), len(self.all_masses)))
        for i in range(len(self.all_redshifts)):
            for k in range(len(self.all_masses)):
                dmo_val = dmo_dictionary_with_hmf[str(i)][str(k)]["dn_dlog10m"]
                hyd_val = hyd_dictionary_with_hmf[str(i)][str(k)]["dn_dlog10m"]
                if hyd_val == 0 or dmo_val == 0:
                    dmo_hmfs[i, k] = 1
                    hyd_hmfs[i, k] = 1
                else:
                    dmo_hmfs[i, k] = dmo_val
                    hyd_hmfs[i, k] = hyd_val

        self.hmf_ratios = hyd_hmfs / dmo_hmfs
