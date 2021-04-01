import time

import pandas as pd

from experiment_utils.entities import Dataset
from experiment_utils.data_manager import preprocessing


from InstanceSelection.DROP2RE import DROP2RE

data_path = '/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/Fast/yacht.dat'
test_files = '/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/Fast/yacht.dat'

# Load dataset
dataset = pd.read_csv(data_path, sep=",", header=None)
# Normalize columns for x and y data
(x, y) = preprocessing(dataset, dataset.shape[1] - 1)

start_time = time.time()
print(DROP2RE(x, y))
print("--- %s seconds ---" % (time.time() - start_time))