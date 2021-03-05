import os

import numpy as np
import pandas as pd


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
    target = df.iloc[:, target_idx].values

    #print(df.info(), data.info(), target.info())
    
    # Apply MinMax norm on every column 
    data = np.apply_along_axis(min_max_norm, 0, data.values)

    return(data, target)


def subsample_file(target_file, result_file, t_file_size = 0.5, rs=30):

    # Load dataset
    dataset = pd.read_csv(target_file, sep=",", header=None)

    # Get the stats of the file
    file_stats = os.stat(target_file)
    # Get the size of the file in MB
    file_size = file_stats.st_size / (1024 * 1024)

    # For reproducibility
    #np.random.seed(rs)
    # Remember the target file size must be given in MB
    #res = dataset.loc[np.random.permutation(dataset.index)[:int((t_file_size / file_size) * len(dataset))]]
    res = dataset.sample(frac=(t_file_size / file_size), random_state=rs)
    #res = res.astype({c: np.float32 for c in res.select_dtypes(include='float64').columns})
    res.to_csv(result_file, index=False)

    #print(result_file, res.info(), pd.read_csv(result_file, sep=",").info())
    #print("Files len:", len(dataset), len(res))
    #print("Files size:", file_size, os.stat(result_file).st_size / (1024 * 1024))
    #print(dataset.head())
    a = np.where(np.isfinite(res.values))
    b = list(zip(a[0], a[1]))[:5]
    c = np.unique(a[0])
    new_df = res.drop(res.index[c])
    print(result_file, len(c), len(res.columns), res.info())
    #print(result_file.split('/')[-1], '\n\t', b, '\n\t', res.values[a], '\n\t', np.where(np.isfinite(new_df.values)))
    #input()