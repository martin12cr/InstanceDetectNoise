import os, sys
import numpy as np

path = os.path.abspath("../")
sys.path.append(path)    

from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression


"""
    This function implements a mutual information algorithm in order to
    detect noisy labels

    @params:
        x       Numpy array containing all the atributes of the sample,
                has shape (num_samples, num_atributes)
        y       Numpy array containing all the regression values of each sample
                has shape (num_samples)
        k       Number of neightbors to use in the algorithm
"""
def mutual_info(x, y, k):

    # Fit the KNN model for the data, by default uses the Minkowski distance 
    # with p = 2, wich is equal to the euclidean distance
    nn_model = NearestNeighbors(n_neighbors=(k + 1)).fit(x)
	# Get the knn
	knn = nn_model.kneighbors(x, return_distance=False)

	# Len x
	lenx = len(x)

    with tqdm(total=lenx, file=sys.stdout) as pbar:
        # For each instance
        for i in range(lenx):
			
			# Calc the MI score without the i-th value
			mi = mutual_info_regression(np.delete(x, i, axis=0))
			



            # Update progress bar
            pbar.update(1)

