import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use("../colibre-stylesheer.mplsht")
import pickle
from scipy.optimize import curve_fit
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

cycle[3] = cycle[5]

def log_log_normal(x,A,mu,sigma):
    return A*np.exp(-1*((np.log10(x)-mu)**2/sigma**2))

fig = plt.figure(figsize=(3.321, 3.321))
iax1 = fig.add_subplot()

with open("../make_hbt_pickles/hist_and_med_L1000N1800_Y.pkl", "rb") as f:
    all_scatter = pickle.load(f)

redshifts = np.zeros(41)
masses = np.zeros(36)
sigs = []
midpoint = -5
big_array = np.zeros(800)

for redshift_index in range(41):
    current_scatter = all_scatter[str(redshift_index)]
    redshifts[redshift_index] = current_scatter["z"]
    temp_list = []
    for mass_index in range(36):
        masses[mass_index] = current_scatter[str(mass_index)]["Mass_cut"]
        print(np.log10(masses[mass_index]))
        c_ys = current_scatter[str(mass_index)]["bins"]
        counts = current_scatter[str(mass_index)]["hist"]
        med = current_scatter[str(mass_index)]["median"]
        cy_hist = counts / (np.sum(counts / np.mean(np.diff(np.log10(c_ys)))))
        if masses[mass_index] < 2e14 or np.sum(counts) <= 3:
            continue
        lin_cy, b = curve_fit(
            log_log_normal,
            c_ys,
            cy_hist,
            p0=(
                np.max(counts) / (np.sum(cy_hist / np.mean(np.diff(np.log10(c_ys))))),
               np.log10(med),
               0.075,
            ),
        )
        bin_shift = int(np.round((np.log10(med)-midpoint)/(np.mean(np.diff(np.log10(c_ys))))))
        big_array = big_array + np.roll(cy_hist,-1*bin_shift)
        sigs.append(lin_cy[2]**2)
        print("cy fit",np.log10(masses[mass_index]), lin_cy)

print(np.sqrt(np.mean(sigs)))
rebinned_arr = big_array
plt.plot(c_ys / 1e-5 ,rebinned_arr/len(sigs),label=r"Sim $M_{\rm 500c}$>$2\times10^{14}$ M$_{\odot}$")
plt.plot(c_ys / 1e-5 ,log_log_normal(c_ys,0.73e-3,-5,np.sqrt(np.mean(sigs))),label="LN fit")
plt.plot(c_ys / 1e-5 ,log_log_normal(c_ys,0.73e-3,-5,0.075),label="LN fit Planck")
plt.loglog()
plt.ylim(0.101e-6,2e-3)
plt.xlim(10**-0.5,10**0.5)
plt.xlabel(r"Compton-Y [Mpc$^{-2}$]")
plt.ylabel("PDF")
plt.legend()
plt.savefig("scat_test.png")
