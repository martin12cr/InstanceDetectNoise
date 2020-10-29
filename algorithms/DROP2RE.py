
import os, sys
import numpy as np

path = os.path.abspath("../")
sys.path.append(path)    

from tqdm import tqdm

from alg_utils.regression_models import train_regresor

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors


"""
    This function implements the DROP2-RE algorithm, wich is an adaptation 
    to regression of DROP2-RE by using error accumulation

    @params:
        x       Numpy array containing all the atributes of the sample,
                has shape (num_samples, num_atributes)
        y       Numpy array containing all the regression values of each sample
                has shape (num_samples)
        k       Number of neightbors to use in the algorithm, for the sklearn model
                to train, its mandatory k >= 11
"""
# Usually the regression models dont converge so ignore the warnings
@ignore_warnings(category=ConvergenceWarning)
def DROP2RE(x, y, k):
    
    # Fit the KNN model for the data, by default uses the Minkowski distance 
    # with p = 2, wich is equal to the euclidean distance
    nn_model = NearestNeighbors(n_neighbors=(k + 1)).fit(x)

    # Add the forbidden idxs to avoid adding them when training the knn
    forbidden_idxs = []


    with tqdm(total=len(x), file=sys.stdout) as pbar:
        # For each instance
        for i in range(len(x)):
            # Get the knn
            knn = nn_model.kneighbors(x[i].reshape(1, -1), return_distance=False)
            # The 0 value in each row is the original, and the 1: are the knn
            knn = [np.array(x[knn])[0], np.array(y[knn])[0]]

            # Instantiate the error with and without the instance
            eWith = 0 
            eWithout = 0

            # Train a model using the instance and another without it 
            instance_model = train_regresor(knn[0], knn[1], test_partition=False)
            instanceless_model = train_regresor(knn[0][1:], knn[1][1:], test_partition=False)

            # Get the total error
            for i in range(1, len(knn[0])):
                # Get the neighbor
                neighbor = knn[0][i].reshape(1, -1)
                # Accumulate error
                eWith += np.abs(instance_model.predict(neighbor) - knn[1][i])
                eWithout += np.abs(instanceless_model.predict(neighbor) - knn[1][i])

            # Check if the error with the instance is worse
            if(eWith > eWithout):
                # Add to the forbidden idxs
                forbidden_idxs.append(i)
                # Re-fit the knn model without the instance
                nn_model = NearestNeighbors(n_neighbors=(k + 1)).fit(np.delete(x, forbidden_idxs, axis=0)) 
            
            # Update progress bar
            pbar.update(1)

    # Sumary
    print('Deleted a total of ', len(forbidden_idxs), ' samples.')

    # TODO HACER QUE EL forbidden_idxs SEA UNA LISTA DE 1 Y 0 EN EL QUE 1 ES UNA INSTANCIA RIUIDOSA

    # Return the clean (x,y)
    return (np.delete(x, forbidden_idxs, axis=0), np.delete(y, forbidden_idxs, axis=0 ))
