import sys, os

import pandas as pd
import numpy as np

from multiprocessing import Pool

sys.path.append('/home/emunoz/InstanceDetectNoise/modules/') # path in KABRE 
#sys.path.append('/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/modules/')


from experiment_utils.entities import Algorithm
from experiment_utils.pack_params import pack_algorithms
from experiment_utils.experiment import experiment_on_data, one_test
from InstanceSelection import RegENN01, RegENN03, RegENN03Wei3, RegBAG, DiscENN, DISKR

# # PATH AND PARAMETERS 

############################## PATH DATA FRAMES ########################################

#data_path = "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/"

#results_path = "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/results/"


data_path = "/home/emunoz/InstanceDetectNoise/Dataset/"
results_path = "/home/emunoz/InstanceDetectNoise/results/"

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
algorithms = {#"RegENN03Wei3": RegENN03Wei3,    # NO FUNCA EN DIAMONDS
            "RegENN01": RegENN01,
            #"RegENN03": RegENN03,              # NO FUNCA EN conductivity
            "DiscENN": DiscENN,
            #"RegBAG": RegBAG,
            #"DISKR": DISKR                     # NO FUNCA EN conductivity
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
i = 0
files = ['laser.dat', 'concrete.dat', 'plastic.dat', 
        'machineCPU.dat', 'autoMPG6.dat', 'diamonds.dat', 
        'california.dat', 'stock.dat', 'mv.dat', 'bike2.dat', 
        'autoMPG8.dat', 'ailerons.dat', 'mortgage.dat', 
        'airfoil.dat', 'CASP.dat', 'conductivty.dat', 'ele-2.dat', 
        'dee.dat', 'house.dat', 'delta_elv.dat', 'Sydney.dat', 
        'metro.dat', 'ANACALT.dat', 'ele-1.dat', 'elevators.dat', 
        'ccpp.dat', 'treasury.dat', 'pole.dat', 'delta_ail.dat', 
        'ccs.dat', 'parkinson.dat', 'wankara.dat', 'friedman.dat', 
        'forestFires.dat', 'realState.dat', 'instanbul.dat', 
        'puma32h.dat', 'wizmir.dat', 'LinkObama.dat', 'bike.dat', 
        'yacht.dat', 'quake.dat', 'energy.dat', 'abalone.dat', 
        'compactiv.dat', 'transcoding.dat', 'electrical.dat', 
        'data.dat', 'baseball.dat', 'qsar.dat']

#files = ['conductivty.dat']

l = len(files)
# Linear execution
for dataset in files[i:]:

    #print(i,'/', l,"\t", dataset)
    #i += 1

    # Run experiments on the dataset seq
    data_results_df = experiment_on_data(alg_params, data_path + dataset , noise_params)
    #one_test(alg_params, data_path + dataset , noise_params)

    # Write to CSV
    #data_results_df.to_csv(results_path + dataset.split('.')[0] + ".csv", index=False)

"""
# Perpare the experiment parameter matrix
exp_params = [(algorithms, data_path + dataset , exp_noise_params) for dataset in os.listdir(data_path)]

# Start with the number of processors
with Pool() as p:

    p.starmap()

"""
