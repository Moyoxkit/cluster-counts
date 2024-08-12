from get_distributions import selections
import numpy as np

dict_folder = "./"

out_list_catalogues = np.array(
    [
        dict_folder + "hist_and_med_L1000N1800_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_WEAK_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONG_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONGER_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONGEST_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_JETS_Y.pkl",
        dict_folder + "hist_and_med_L1000N1800_STRONG_JETS_Y.pkl",
    ]
)

pre_folder = "/snap8/scratch/dp004/dc-mcgi1/roi/minimal/Runs/L1000N1800/"

in_list_catalogues = np.array(
    [
        pre_folder + "HYDRO_FIDUCIAL/SOAP_uncompressed/HBTplus/",
     	pre_folder + "HYDRO_WEAK_AGN/SOAP_uncompressed/HBTplus/",
     	pre_folder + "HYDRO_STRONG_AGN/SOAP_uncompressed/HBTplus/",
     	pre_folder + "HYDRO_STRONGER_AGN/SOAP_uncompressed/HBTplus/",
     	pre_folder + "HYDRO_STRONGEST_AGN/SOAP_uncompressed/HBTplus/",
     	pre_folder + "HYDRO_JETS_published/SOAP_uncompressed/HBTplus/",
     	pre_folder + "HYDRO_STRONG_JETS_published/SOAP_uncompressed/HBTplus/",
    ]
)

for i in range(7):
    banana = selections(in_list_catalogues[i], out_list_catalogues[i])
    banana.get_hist()
