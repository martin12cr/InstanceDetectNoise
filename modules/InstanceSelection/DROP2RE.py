import numpy as np
from numpy.core.records import array

#from tqdm import tqdm


from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor




"""
    This function gets the knn of a given point, it also uses the point index
    in order to remove it frrom the nn list 

    @params:
        model       KNN trained model
        point       The value we want to get the knn
        point_idx   The index of the inferenced value
"""
def get_knn(model, point, point_idx):

    knn_idxs = model.kneighbors(point.reshape(1, -1), return_distance=False)[0]

    # If the xi value is in the knn then remove it
    if(point_idx in knn_idxs):
        knn_idxs = np.delete(knn_idxs, np.where(knn_idxs == point_idx), axis=0)
    # If it isn't then select only the k+1 nn
    else: 
        knn_idxs = knn_idxs[:-1]

    return knn_idxs




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
def DROP2RE(x, y, k=9):
    
    # Fit the KNN model for the data, by default uses the Minkowski distance 
    # with p = 2, wich is equal to the euclidean distance
    # We select k + 2 NN beacause tha algorithm suggest using K + 1 and in this
    # implementation of KNN the value itself is considered a neighbor
    knn_model = NearestNeighbors(n_neighbors=(k + 2)).fit(x)

    # Create the KNN regresion models 
    knn_reg_model_w = KNeighborsRegressor(n_neighbors=(k))
    knn_reg_model_wo = KNeighborsRegressor(n_neighbors=(k))

    # Add the forbidden idxs to avoid adding them when training the knn
    forbidden_idxs = []

    # Create the associate list and the knn list
    associates = []
    knn = []
    # We have to initialize it this way to force diferent list objects into 
    # each space and avoid having the same reference into every sapce
    for i in range(len(x)): 
        associates.append([])
        knn.append([])

    # Find the associates of each sample
    for i in range(len(x)):

        # Get the k+1 nn
        knn[i] = get_knn(knn_model, x[i], i)

        # Add x index to its neighbors list of associates, since 
        # we are only using KNN and calcualating K+1NN, only add 
        # xi as an associate to the KNN and leave the extra NN
        # as an adition when needed to train the models
        for n_idx in knn[i][:-1]: associates[n_idx].append(i)


    # For each instance 
    for i in range(len(x)):

        # Instantiate the error with and without the instance
        eWithout = 0
        eWith = 0

        # Iterate through each xi associate, train one model using the knn w/o
        # the instance xiand another with the instance
        for j in range(len(associates[i])):

            # Get the associate idx
            associate_idx = associates[i][j]
            # Get the knn indexes
            knn_wo_idxs = np.delete(knn[associate_idx],
                                np.where(knn[associate_idx] == i), 
                                axis=0)
            knn_w_idxs = knn[associate_idx][:-1]

            # Get the idxs of the xi associates, select the aj associate
            # indx and get the idx of its knn, then get the values 
            # When working without the instance, remove it from the
            # knn list of indexes
            aj_knn_wo_x = x[knn_wo_idxs, :]
            aj_knn_wo_y = y[knn_wo_idxs]
            aj_knn_w_x = x[knn_w_idxs, :]
            aj_knn_w_y = y[knn_w_idxs]

            # Train a model without the instance
            knn_reg_model_wo.fit(aj_knn_wo_x, aj_knn_wo_y)
            # Train a model with the instance
            knn_reg_model_w.fit(aj_knn_w_x, aj_knn_w_y)

            # Accumulate error
            eWithout += np.abs(knn_reg_model_wo.predict(x[associate_idx].reshape(1, -1)) - y[associate_idx])
            eWith += np.abs(knn_reg_model_w.predict(x[associate_idx].reshape(1, -1)) - y[associate_idx])


        # Check if the error with the instance is worse
        if(eWith > eWithout):

            # Add to the forbidden idxs
            forbidden_idxs.append(i)

            # Remove x from its associates list of nn
            for k in range(len(associates[i])):

                # Get the associate index
                associate_idx = associates[i][k]

                # Re-fit the knn model without the instance
                knn_model.fit(np.delete(x, forbidden_idxs, axis=0)) 

                # Find the new nearest neigbors for the associate
                knn[associate_idx] = get_knn(knn_model, x[associate_idx], associate_idx)

                # Verify that every ak neighboor has ak as an associate 
                for n_idx in knn[associate_idx][:-1]: 
                    # If the nn does not have ak as an associate
                    if(not (associate_idx in associates[n_idx])):
                        # Add it
                        associates[n_idx].append(associate_idx)
        
        # Update progress bar
        #pbar.update(1)

    # Sumary
    #print("Deleted a total of ", len(forbidden_idxs), " out of ", len(x), " samples.")

    # Return a binary list with a 1 on the noisy samples idx
    return np.sum(np.equal(np.arange(len(x)), np.array(forbidden_idxs)[:,np.newaxis]), 0)
