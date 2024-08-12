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
from hmf import MassFunction
import MiraTitanHMFemulator
import commah
from colossus.cosmology import cosmology as colo_cosmo
from colossus.halo import concentration as colo_conc


class cluster_cosmology_model:
    """
    Class that has all the tools for creating simple cluster count models.
    """

    def __init__(
        self,
        FLAMINGO_info,
        y_cut,
        cosmo_info=None,
        fit_model=None,
        power_law_meds=False,
        log_normal_scatter=False,
        true_halo_mass_function=False,
        mira_titan=False,
        log_normal_lognsigy=0.075,
        power_law_args=(0.79169079, 1.67645383, 0.66),
        use_hydro_hmf_ratio=False,
    ):
        """
        Initialises the mass definition used for the halo mass function,
        initializes the default FLAMINGO cosmology and sets up the interpolators
        for the FLAMINGO medians and scatter
        """

        self.FLAMINGO_functions = FLAMINGO_info
        if cosmo_info == None:
            self.set_flamingo_cosmology()
        else:
            self.set_cosmology(cosmo_info)

        power_law_args = [
            power_law_args[0],
            power_law_args[1],
            power_law_args[2],
            log_normal_lognsigy,
        ]
        self.FLAMINGO_functions.init_other_interpolators(
            self.astropy_cosmology, power_law_args
        )

        self.mira_titan_hmf = MiraTitanHMFemulator.Emulator()

        if mira_titan == False and true_halo_mass_function == False:
            self.hmf = MassFunction(
                cosmo_model=self.astropy_cosmology,
                mdef_model="SOCritical",
                mdef_params={"overdensity": 500},
                dlog10m=0.001,
                hmf_model=fit_model,
                transfer_model="EH",
                Mmin=10.5,
                Mmax=17,
                sigma_8=self.cosmological_parameters["sigma_8"],
                n=self.cosmological_parameters["n_s"],
            )

        self.init_number_counts_sz(
            y_cut,
            power_law_meds=power_law_meds,
            log_normal_scatter=log_normal_scatter,
            true_halo_mass_function=true_halo_mass_function,
            mira_titan=mira_titan,
            log_normal_lognsigy=log_normal_lognsigy,
            power_law_args=power_law_args,
            use_hydro_hmf_ratio=use_hydro_hmf_ratio,
        )

    def mass_translator(self, M500s, z, cosmology):
        """
        Function that converts and array of M500s to an array of M200s at
        a given cosmology and z.
        """

        def find_R500(R_200, c):
            conc_Y_c = np.log(1 + c) - c / (1 + c)
            delta_c = 200 * c**3 / (3 * conc_Y_c)
            R_s = R_200 / c
            rmax = np.logspace(-6, 1, 100)
            den_enclosed_in_crits_d = (3 * delta_c * (R_s / rmax) ** 3) * (
                np.log(((R_s + rmax) / R_s)) - (rmax / (R_s + rmax))
            )
            den_to_r = interp1d(den_enclosed_in_crits_d, rmax)
            return den_to_r(500)

        M200s = np.logspace(12, 16.5, 45)

        concs = colo_conc.modelDiemer19(
            M200s / (self.cosmological_parameters["H_0"] / 100), z, statistic="mean"
        )
        concs = concs[0]
        crit_den = (
            self.astropy_cosmology.critical_density(z).to(u.Msun / (u.Mpc**3)).value
        )
        R_200s = (M200s / (4 * np.pi * 200 * crit_den / 3)) ** (1 / 3)
        int_M500s = np.zeros(len(R_200s))
        for i in range(len(M200s)):
            R_500 = find_R500(R_200s[i], concs[i])
            int_M500s[i] = 4 * np.pi * 500 * crit_den * R_500**3 / 3

        M500_to_M200 = interp1d(int_M500s, M200s)
        return M500_to_M200(M500s)

    def set_cosmology(self, cosmological_parameters, mira_titan=False):
        """
        Sets the cosmology used for the halo mass function and the
        differential volume element
        """
        m_nu = [cosmological_parameters["m_nu"], 0.00, 0.00] * u.eV
        self.astropy_cosmology = FlatLambdaCDM(
            H0=cosmological_parameters["H_0"],
            Om0=cosmological_parameters["Omega_m"],
            m_nu=m_nu,
            Ob0=cosmological_parameters["Omega_b"],
            Tcmb0=2.725,
        )
        self.cosmological_parameters = cosmological_parameters

        self.mira_titan_cosmology = {
            "Ommh2": cosmological_parameters["Omega_m"]
            * (cosmological_parameters["H_0"] / 100) ** 2,
            "Ombh2": cosmological_parameters["Omega_b"]
            * (cosmological_parameters["H_0"] / 100) ** 2,
            "Omnuh2": cosmological_parameters["m_nu"] * (0.01 / 0.94),
            "n_s": cosmological_parameters["n_s"],
            "h": cosmological_parameters["H_0"] / 100,
            "w_0": cosmological_parameters["w_0"],
            "w_a": cosmological_parameters["w_a"],
            "sigma_8": cosmological_parameters["sigma_8"],
        }

        self.commah_cosmology = {
            "omega_M_0": cosmological_parameters["Omega_m"],
            "omega_b_0": cosmological_parameters["Omega_b"],
            "omega_lambda_0": 1 - cosmological_parameters["Omega_m"],
            "omega_n_0": cosmological_parameters["m_nu"] * (0.01 / 0.94),
            "n": cosmological_parameters["n_s"],
            "h": cosmological_parameters["H_0"] / 100,
            "w_0": cosmological_parameters["w_0"],
            "w_a": cosmological_parameters["w_a"],
            "sigma_8": cosmological_parameters["sigma_8"],
        }

        bla = colo_cosmo.fromAstropy(
            self.astropy_cosmology,
            cosmological_parameters["sigma_8"],
            cosmological_parameters["n_s"],
            cosmo_name="colo_cosmo",
        )
        colo_cosmo.setCurrent(bla)

        # Update the hmf cosmology if it has been already initialized
        if hasattr(self, "hmf") and mira_titan == False:
            self.hmf.update(
                cosmo_model=self.astropy_cosmology,
                sigma_8=self.cosmological_parameters["sigma_8"],
                n=self.cosmological_parameters["n_s"],
            )

    def get_cosmology(self):
        return self.cosmological_parameters

    def set_flamingo_cosmology(self):
        """
        Sets all the cosmology objects to the fiducial FLAMINGO cosmology
        and initialises the halo mass function at this cosmology
        """
        self.flamingo_cosmology = {
            "Omega_m": 0.306,
            "Omega_b": 0.0486,
            "m_nu": 0.06,
            "H_0": 68.1,
            "sigma_8": 0.807,
            "n_s": 0.967,
            "w_0": -1,
            "w_a": 0,
        }
        self.set_cosmology(self.flamingo_cosmology)

    def differential_volume(self, z, solid_angle=4 * np.pi):
        """
        Returns the dV element at a input redshift for the reference cosmology in Mpc^3

        Parameters:

        z           : redshift
        solid_angle : Angle subtended on the sky, default is 4pi, the full sky in sr
        """

        return (
            (
                self.astropy_cosmology.differential_comoving_volume(z)
                * solid_angle
                * u.sr
            )
            .to(u.Mpc**3)
            .value
        )

    def halo_mass_function(self, M500s, z, mira_titan=False):
        """
        Returns the value of the halo mass function for a given mass and redshift

        Parameters:
        M500s : Mass in terms of M500crit
        z     : Redshift
        """

        little_h = self.astropy_cosmology.H(0).value / 100
        if mira_titan:
            if z > 2.02:
                return np.zeros(len(M500s))

            masses_to_interpolate = np.logspace(13, 16, 1000) / little_h
            numden_to_interpolate = (
                self.mira_titan_hmf.predict(
                    self.mira_titan_cosmology,
                    z,
                    masses_to_interpolate * little_h,
                    get_errors=False,
                )[0][0]
                * little_h**3
                * np.log(10)
            )
            dn_dlog10m_interpolator = interp1d(
                masses_to_interpolate,
                numden_to_interpolate,
                fill_value=0,
                bounds_error=False,
            )

            mass_limit_200 = self.mass_translator(M500s, z, self.commah_cosmology)

            #            for index, mass_limit_200_i in enumerate(mass_limit_200):
            #                mass_limit_200[index] = np.mean(
            #                    10 ** np.random.normal(np.log10(mass_limit_200_i), 0.16, 1000)
            #                )

            return dn_dlog10m_interpolator(mass_limit_200)

        else:
            self.hmf.update(z=z)
            dn_dlog10m_interpolator = interp1d(
                self.hmf.m / little_h, self.hmf.dndlog10m * little_h**3
            )
            return dn_dlog10m_interpolator(M500s)

    def init_number_counts_sz(
        self,
        y_cut,
        power_law_meds=False,
        log_normal_scatter=False,
        true_halo_mass_function=False,
        mira_titan=False,
        log_normal_lognsigy=0.075,
        power_law_args=(0.79169079, 1.67645383, 0.66),
        use_hydro_hmf_ratio=False,
    ):
        """
        Calculate the number counts using simple integration. Integrates
        the expected number counts between a lower redshift z_low and a higher
        redshift z_high for a survey with an SZ cut at y_cut. By default it
        uses the medians and scatter from the input FLAMINGO simulation but has
        options to use a power-law scaling relation instead.

        Parameters:

        z_low    : redshift at which to start integration
        z_high   : redshift to integrate to
        y_cut    : Compton Y cut that defines the selection
        power_law_meds : Boolean, use a power law for medians?
        log_normal_scatter: Boolean, use log normal scatter?
        true_halo_mass_function: Boolean, use the HMF from FLAMINGO?
        log_normal_lognsigy: log normal scatter in dex. Only used when log_normal_scatter=True
        use_hydro_hmf_ratio: Apply the ratio of DMO to hydro to alter the HMF
        """

        # Check if we want both log normal scatter and a power law
        power_law_and_log_normal = False
        if log_normal_scatter and power_law_meds:
            power_law_and_log_normal = True

        to_integrate = np.zeros(
            (
                len(self.FLAMINGO_functions.all_redshifts),
                len(self.FLAMINGO_functions.all_masses),
            )
        )
        for red_ind in range(len(self.FLAMINGO_functions.all_redshifts)):
            if true_halo_mass_function:
                number_densities = self.FLAMINGO_functions.flamingo_number_densities[
                    red_ind, :
                ]
            else:
                number_densities = self.halo_mass_function(
                    self.FLAMINGO_functions.all_masses,
                    self.FLAMINGO_functions.all_redshifts[red_ind],
                    mira_titan=mira_titan,
                )
            if use_hydro_hmf_ratio:
                number_densities = (
                    self.FLAMINGO_functions.hmf_ratios[red_ind, :] * number_densities
                )

            volume = self.differential_volume(
                self.FLAMINGO_functions.all_redshifts[red_ind]
            )
            for mass_ind in range(len(self.FLAMINGO_functions.all_masses)):
                # Start by checking if we want power law + log normal as this overides all other options
                if power_law_and_log_normal:
                    halo_fraction = self.FLAMINGO_functions.scatter_interpolators_LN_PL[
                        red_ind
                    ][mass_ind](np.log10(y_cut))
                elif power_law_meds:
                    halo_fraction = self.FLAMINGO_functions.scatter_interpolators_PL[
                        red_ind
                    ][mass_ind](np.log10(y_cut))
                elif log_normal_scatter:
                    halo_fraction = self.FLAMINGO_functions.scatter_interpolators_LN[
                        red_ind
                    ][mass_ind](np.log10(y_cut))
                else:
                    halo_fraction = self.FLAMINGO_functions.scatter_interpolators[
                        red_ind
                    ][mass_ind](np.log10(y_cut))
                to_integrate[red_ind, mass_ind] = (
                    halo_fraction * number_densities[mass_ind] * volume
                )

        # first integral over mass is performed by summing out one dimension of the array
        # No need for interpolation as we always want to include all halo masses
        integrated_with_m500 = np.sum(to_integrate, axis=1) * np.mean(
            np.diff(np.log10(self.FLAMINGO_functions.all_masses))
        )
        # We need to have a continuos answer in z so we interpolate the resulting array
        self.func_for_int = interp1d(
            self.FLAMINGO_functions.all_redshifts, integrated_with_m500
        )

    def number_counts_sz(self, z_low, z_high):
        """
        Use the previously initialised z_integral to calculate the actual number counts
        """
        # We can then do the integral over a continuous integral
        return quad(self.func_for_int, z_low, z_high)
