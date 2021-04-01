
import numpy as np

from scipy.stats import rankdata



# Function to get the euclidean distance
def distan(x,data):
    res=(x-data)**2
    return(np.sum(res,1)**0.5)

# Function to get position from smallest to bigger at row
def idxSize(x): 
    idx = rankdata(x, method='ordinal')
    return(idx)

# Function to get indexes of smallest values. 
def idxSmall(x, k): 
    idx = np.argpartition(x,k+1)
    return (idx)

# Function to get mean and standart deviation.
def stat(x, response): 
    m = np.mean(response[x])
    s = np.std(response[x])
    return (m,s)
