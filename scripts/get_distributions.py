import numpy as np
import unyt
import matplotlib.pyplot as plt
from tqdm import tqdm
import unyt
import swiftsimio as sw
import h5py as h5
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--soap", type=str)
args = parser.parse_args()


class selections:
    """
    class that allows you to do a bunch of fun selection things
    """

    def __init__(self, SOAP_folder, output_file) -> None:
        self.SOAP_dir = SOAP_folder
        self.output_file = output_file
        self.Nbins = 40
        self.has_figure = False
        self.cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.snapshots = np.arange(77, 77 - 61, -1)
        self.snapshot_z = np.linspace(0, 3, 61)

    def load_catalogue(self, snapshot_number, z_of_snapshot):
        self.catalogue = sw.load(
            self.SOAP_dir + "/halo_properties_00" + str(snapshot_number) + ".hdf5"
        )
        self.mass_500s = self.catalogue.spherical_overdensity_500_crit.total_mass.to(
            unyt.Msun
        ).value
        #        print(len(self.mass_500s[self.mass_500s > 5e14]))
        #        self.to_bin = self.catalogue["SO/500_crit/XRayLuminosityWithoutRecentAGNHeating"][:][:, 0]
        self.to_bin = (
            (
                self.catalogue.spherical_overdensity_500_crit.compton_ywithout_recent_agnheating
            )
            .to(unyt.Mpc**2)
            .value
        )
        self.z = z_of_snapshot

    def get_hist(self):
        output_dict = {}
        for snapshot_ind in tqdm(range(41)):  # 61 is z = 3
            print(self.snapshots[snapshot_ind])
            self.load_catalogue(
                self.snapshots[snapshot_ind],
                z_of_snapshot=self.snapshot_z[snapshot_ind],
            )
            dict_for_snap = {}
            dict_for_snap["z"] = self.snapshot_z[snapshot_ind]
            mass_limits = np.logspace(12, 15.5, 36)
            mass_ind = 0
            for mass_limit in tqdm(mass_limits, leave=False):
                fixed_mass_selection = (self.mass_500s > mass_limit * 10 ** (-0.05)) & (
                    self.mass_500s < mass_limit * 10 ** (0.05)
                )
                hist_dict = {}
                hist_dict["Mass_cut"] = mass_limit
                #                x_bins = np.logspace(34,50,640) #For z=0 X-ray
                x_bins = np.logspace(-10, -2, 801)  # For z=0 SZ
                x_hist, _ = np.histogram(self.to_bin[fixed_mass_selection], x_bins)
                hist_dict["bins"] = (x_bins[1:] + x_bins[:-1]) / 2
                hist_dict["hist"] = x_hist
                hist_dict["median"] = np.median(self.to_bin[fixed_mass_selection])
                hist_dict["dn_dlog10m"] = len(self.to_bin[fixed_mass_selection]) / (
                    1000**3 * 0.1
                )
                dict_for_snap[str(mass_ind)] = hist_dict
                mass_ind += 1
            output_dict[str(snapshot_ind)] = dict_for_snap

        f = open(self.output_file, "wb")
        print(output_dict)
        pickle.dump(output_dict, f)
        f.close()
