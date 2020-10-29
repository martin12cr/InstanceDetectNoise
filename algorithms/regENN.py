import numpy as np

from sklearn.neighbors import NearestNeighbors


"""
    This function implements the RegENN algorithm 
    (Edited Nearest Neighbor for regression using a threshold)

    @params:
        x       Numpy array containing all the atributes of the sample,
                has shape (num_samples, num_atributes)
        y       Numpy array containing all the regression values of each sample
                has shape (num_samples)
        alpha   Controls how the threshold is calculated from the std
        k       Number of neightbors to train the model
        model   Regresion model trained without the samples in T

"""
def regENN(x, y, alpha, k, model):

    # First verify the data is consistent
    assert len(x) == len(y), "'x' and 'y' must have the same number of samples"

    # Container for the index of the selected instances
    sx = []
    sy = []

    # Fit the KNN model for the data, by default uses the Minkowski distance 
    # with p = 2, wich is equal to the euclidean distance
    nn_model = NearestNeighbors(n_neighbors=k).fit(x)
    
    # For each sample in the training set 
    for i in range(len(x)):
        # Get the sample xi
        xi = x[i].reshape(1, -1)
        print(xi.shape)
        yi = y[i]
        # Make a prediction of xi
        pred_y = model.predict(xi)
        # Get the KNN of xi, this returns the distances and the indexes
        idxs = nn_model.kneighbors(xi, return_distance=False)
        print(idxs)
        # Calc the standard deviation of the neightbors
        theta = alpha * np.std(y[idxs])

        print(y[idxs])
        print('std: ', np.std(y[idxs]))

        print('Predicted y: ', pred_y,'\tReal y: ', yi, 
            'Difference: ', np.abs(yi - pred_y), '\tTheta: ', theta)
        print('np.abs(yi - pred_y) < theta: ', np.abs(yi - pred_y) < theta)
        
        # The sample is considered valid if the MAE between
        # the prediction and the original value is smaller
        # than the std of the k nearest
        if(np.abs(yi - pred_y) < theta):
            sx.append([xi])
            sy.append([yi])

        input('----------------------------------------------------')

    # Return the results
    return np.array(sx), np.array(sy)

