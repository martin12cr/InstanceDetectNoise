The main.py allows executing the process indicated in figure 1 of the article. This file generates a CSV with the performance metrics of an instance selection algorithm applied to a set of datasets. Before executing main.py the following must you must defined within the file 
data_path: path of datasets

#results_path: path to save the results (csv file)
#perc_noisy_instances: vector with the noise percentages that will use it
#perc_added_noise: vector with the magnitudes percentages that will use it
#isFiltered: vector with the value of 1 if IS algorithm is applied, or 0 if it is not applied
#k: number of neighbors of KNN
regs_alpha: vector  with alpha parameter values to be used in RegENN1, RegENN2, RegENN3
diskr_alpha: vector alpha values to be used in DISKR  
num_bins: vector with number of bins to be used in DiscENN
algorithms: Dictionary with names of IS algorithms that will be used



In the folder named InstaceSelection are the IS algorithms used in the article
In the folder named Datasets are the data files used in the article
