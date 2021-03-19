import warnings, sys, os

import pandas as pd
import numpy as np

from multiprocessing import Pool


# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn




base_path = "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/"

sys.path.append(base_path + "modules/")


from experiment_utils.entities import Algorithm
from experiment_utils.pack_params import pack_algorithms
from experiment_utils.experiment import experiment_on_data, one_test
from InstanceSelection import RegENN01, RegENN03, RegENN03Wei3, RegBAG, DiscENN, DISKR

# # PATH AND PARAMETERS 

############################## PATH DATA FRAMES ########################################

data_path = base_path + "Dataset/diskr/"

results_path = base_path + "results_test/"


#data_path = "/home/emunoz/InstanceDetectNoise/Dataset/Fast/"
#results_path = "/home/emunoz/InstanceDetectNoise/results/"

############################### NOISE PARAMETERS #######################################

# Number of added noisy instances (%)
perc_noisy_instances = np.array([0, 0.1, 0.3, 0.5])
# Percentage of noise added over the instance
perc_added_noise = np.array([0.1, 0.5, 1, 2]) 
# Noise type 
noise_type = np.array(["persize"])
# Filter 1 means yes
isFiltered = np.array([1])

# Concatneate all noise parameters
noise_params = [perc_noisy_instances, perc_added_noise, noise_type, isFiltered]


################################# FUNCTION AND PARAMETERS #################################
# Function parameters for RegENN01,RegENN01, and RegBAG

################################# FUNCTION PARAMETERS #################################

# Kmeans values
k = np.array([9], dtype=np.int)

# Alpha values for Reg algorithms and DISKR
regs_alpha = np.array([0.5, 1, 3, 5])
diskr_alpha = np.array([0.05, 0.10, 0.2, 0.3])

# Number of bags and sample size for bagging
num_bags = np.array([100], dtype=np.int)
sample_size = np.array([1], dtype=np.int)

# Number of bins and strategy for DiscENN
strategy = np.array(['persize'])
num_bins = np.array([2, 3, 4, 5], dtype=np.int)

# Select the algorithms for the experiment
algorithms = {#"RegENN03Wei3": RegENN03Wei3,    
            #"RegENN01": RegENN01,
            #"RegENN03": RegENN03,              
            #"DiscENN": DiscENN,                    # Tira varios errores
            #"RegBAG": RegBAG,                     # No correr
            "DISKR": DISKR                         # Tira varios errores
        }

alg_params = pack_algorithms(algorithms, 
                            k, 
                            regs_alpha, 
                            diskr_alpha, 
                            num_bags, 
                            sample_size, 
                            strategy,
                            num_bins)  

# Todos con k = 9
# Probar DISKR con alpha = [0.05, 0.10, 0.2, 0.3]
# Probar RegENN01, RegENN03, RegENN03Wei3 y RegBAG con alpha = [0.5, 1, 3, 5]
# RegBAG numBag=100, samSize=1 siempre
# Probar DiscENN con bins = [2,3,4,5]     strat = 'uniform'





################################# EXPERIMENT EXECUTION #################################

# Detect available files
files = os.listdir(data_path)
files.sort()

# Do not process finished files
for r in os.listdir(results_path):
    r = r.split('.')[0] + ".dat"
    if (r in files):
        files.remove(r.split('.')[0] + ".dat")

"""files.remove("dee.dat")
files.remove("ele-1.dat")
files.remove("ele-2.dat")
files.remove("energy.dat")
files.remove("forestFires.dat")
files.remove("friedman.dat")
files.remove("machineCPU.dat")
files.remove("mortgage.dat")
files.remove("plastic.dat")
files.remove("realState.dat")
files.remove("stock.dat")
files.remove("treasury.dat")
files.remove("yacht.dat")"""

#files.remove("ailerons.dat")


print(results_path)
print(os.listdir(results_path))
print(files)

l = len(files)
# Linear execution
for dataset in files:

    # Run experiments on the dataset seq
    data_results_df = experiment_on_data(alg_params, data_path + dataset , noise_params)
    #one_test(alg_params, data_path + dataset , noise_params)

    # Write to CSV
    data_results_df.to_csv(results_path + dataset.split('.')[0] + ".csv", index=False)
