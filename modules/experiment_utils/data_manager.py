import os

import pandas as pd
import numpy as np


"""
    MinMax normalñization function on an array x

    x:  1D array target

"""
def min_max_norm(x):

    # Apply MinMax normalization on x
    norm_x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return norm_x


"""
    Function to separate the response from the data and normalize each column of the dataset 

    df:             pd.DatasFrame with the dataset
    target_idx:     Index of the response 

"""
def preprocessing(df, target_idx, rs=30): 
    
    # Sort the df
    df = df.sample(frac=1, random_state=rs)  
    
    # Exclude the target from train_data
    data = df.loc[:, df.columns != target_idx]
    target = df.loc[:, target_idx].values
    
    # Apply MinMax norm on every column 
    data = np.apply_along_axis(min_max_norm, 0, data.values)

    return(data, target)