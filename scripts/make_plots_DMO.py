from get_distributions_DMO import selections

input_locs = [
    "/snap8/scratch/dp004/dc-mcgi1/flamingo_dmo/Runs/L5600N5040/DMO_FIDUCIAL/SOAP_uncompressed/HBTplus",
    "/snap8/scratch/dp004/dc-mcgi1/flamingo_dmo/Runs/L1000N1800/DMO_FIDUCIAL/SOAP_uncompressed/HBTplus/",
]

output_files = [
    "hist_and_med_L5600N5040DMO.pkl",
    "hist_and_med_L1000N1800DMO.pkl",
]

for i in range(1):
    banana = selections(input_locs[i], output_files[i])
    banana.get_hist()
