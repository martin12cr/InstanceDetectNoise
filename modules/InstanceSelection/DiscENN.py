
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer
from imblearn.under_sampling import EditedNearestNeighbours





##################### DiscENN_SIN OPTIMIZADOR ############################

# data=data without response variable
# response= response variable
# strat= method for discretization , we will use uniform
# bins= number of groups to discretize 
# k= number of neihgboors

### OUTPUT
# vector indicating which instances has noisy  in the response variable


def DiscENN(data, response, k=9, kbins_strat='uniform', bins=5): 

    #print("DiscENN")

    k = int(k)
    bins = int(bins)
    
    # Apply ENN, get indexes of selected samples (No noise samples)
    responseTem=np.array(response).reshape(-1, 1)

    # Apply KBins
    kbins = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=kbins_strat)
    response_trans = kbins.fit_transform(responseTem)

    # Apply ENN for categorical response
    ENN = EditedNearestNeighbours(n_neighbors=k,kind_sel='mode', return_indices=True).fit(data,  response_trans)
    selec=ENN.sample(data,  response_trans)[2]

    # get array of all indexes

    todos=np.arange(response.shape[0])

    # get which were not selected 
    NoSelec=np.setdiff1d(todos, selec, assume_unique=False)

    # add code to selected and not selected 
    unos=np.repeat(0, selec.shape)
    ceros=np.repeat(1, NoSelec.shape)
    noise=np.hstack((unos,ceros))
    indices=np.hstack((selec,NoSelec))
    unidas=np.column_stack((indices,noise))

    # get final noise vector 
    noise = unidas[np.argsort(unidas[:, 0])][:,1]
    
    return noise