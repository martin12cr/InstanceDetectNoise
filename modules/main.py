import sys, os

import pandas as pd
import numpy as np

from multiprocessing import Pool

# sys.path.append('/home/msolis/InstanceDetectNoise/modules/') # path in KABRE 
sys.path.append('/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/modules/')

from experiment_utils.entities import Algorithm
from experiment_utils.experiment import experiment_on_data
from InstanceSelection import RegENN01, RegENN03, RegBAG, DiscENN, ENN


# # PATH AND PARAMETERS 

############################## PATH DATA FRAMES ########################################

# RootOr='/work/msolis/data/'  # PATH KABRE 
data_path = "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/"
results_path = "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/results/"

############################### NOISE PARAMETERS #######################################

# Number of added noisy instances (%)
perc_noisy_instances = np.array([0.1, 0.3, 0.5])
# Percentage of noise added over the instance
perc_added_noise = np.array([0.1, 0.5, 1, 2]) 
# Noise type 
noise_type = np.array(["persize"])
# Filter 1 means yes
isFiltered = np.array([1])

# Baseline parameters (0 noise + filter)
baseline_filtered = np.array([0, 0, 1]).reshape(1, -1) 
#baseline_unfiltered = np.array([0, 0, 0]).reshape(1, -1) 

# Matrix of combined parameters for the experiments
exp_noise_params = np.array(np.meshgrid(perc_noisy_instances, perc_added_noise, isFiltered)).T.reshape(-1, 3)
# Append Baseline parameters to the experiment matrix
exp_noise_params = np.append(exp_noise_params, baseline_filtered, axis=0) 
#exp_noise_params = np.append(exp_noise_params, baseline_unfiltered, axis=0) 

noise_params = [perc_noisy_instances, perc_added_noise, noise_type, isFiltered]


################################# FUNCTION AND PARAMETERS #################################
# Function parameters for RegENN01,RegENN01, and RegBAG
k = 9
alphas = np.array([0.5, 1, 3, 5])  # este se llama j en old


################################# FUNCTION DICTIONARY #################################
algorithms = [
    Algorithm("RegENN01", RegENN01),
    Algorithm("RegENN03", RegENN03),
    Algorithm("DiscENN", DiscENN), # In this case we will prove num_bins with:2,3,4,5
    Algorithm("RegBAG", RegBAG)
]







################################# EXPERIMENT EXECUTION #################################

# Linear execution
for dataset in os.listdir(data_path):

    # Run experiments on the dataset seq
    data_results_df = experiment_on_data(algorithms, data_path + dataset , noise_params)

    # Write to CSV
    data_results_df.to_csv(results_path + dataset.split('.')[0] + ".csv", index=False)

    break

"""
# Perpare the experiment parameter matrix
exp_params = [(algorithms, data_path + dataset , exp_noise_params) for dataset in os.listdir(data_path)]

# Start with the number of processors
with Pool() as p:

    p.starmap()

"""
