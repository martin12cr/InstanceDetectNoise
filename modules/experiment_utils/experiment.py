import sys, time

import numpy as np
import pandas as pd

from multiprocessing import Pool

from .entities import Dataset
from .data_manager import preprocessing

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error




# # NOISE FUNCTION 


# NOISE FUNCTION 
#### inputs 


### Outputs
# return two arrays, one with the target variable with noise in some instances, and a dummy vector indicating if the instance is noisy
"""
    Function designed to add percentage of noise to a fixed percentage of the given samples

    target:     Target variable or output variable as vector
    perc:       Percentage of noisy instances
    method:     Noise method, can be 'normal (gaussian)', 'uniform' or 'persize'

    return:
    Two arrays, one with the target variable with noise in some instances, and a dummy vector indicating if the instance is noisy

"""
def add_noise(target, perc, method, mag=0.3, rs=30):    
    
    # Set the random seed for reproduceability
    np.random.seed(rs)
    # Create array with consecutive, random vector and target vector
    arr = np.stack((np.r_[1:target.shape[0] + 1], np.random.rand(target.shape[0]), target), axis=-1)

    # Sort array according to random vector 
    arr = arr[np.argsort(arr[:, 1])]

    #### Generate noise vector

    # number of values
    c1 = int(target.shape[0] * perc)

    # to make the ramdom noise positive or negative
    mul1 = np.random.randint(2, size=c1)
    mul1[mul1 == 0] = -1
    
    if (method == 'normal'):
        # to make the ramdom noise positive or negative
        noise1 = np.random.normal(np.mean(target), np.std(target), c1) * mul1

    elif (method == 'uniform'):
        noise1=np.random.uniform(np.min(target),np.max(target),c1) * mul1

    elif (method == 'persize'):
        if perc > 0:
            noise1 = arr[:, 2][-c1:target.shape[0]] * mag * mul1
        else:
            noise1 = np.zeros(target.shape[0])
            
    
    if (perc>0):    
        # Stack noise vectors
        noise = np.hstack((np.zeros(target.shape[0] - (c1)), noise1))
    else:
        noise = noise1
    
    # stack noise vector in matrix
    arr = np.column_stack((arr, noise))
    
    # Add noise to the target
    agg = arr[:, 2] + arr[:, 3]
    arr = np.column_stack((arr, agg))
    
    # Sort array according to consecutive
    arr = arr[np.argsort(arr[:, 0])]
    
    # Recoding noise in two groups, 1= noise instance, 0= not noise
    noiseAbs = np.abs(arr[:, 3])
    noiseAbs[noiseAbs == 0] = 0
    noiseAbs[noiseAbs > 0] = 1
    noiseAbs[noiseAbs < 0] = 1
    
    return([arr[:,4], noiseAbs])


"""
    Function to excecute a specific noise combination on a given dataset

    dataset:        Dataset class that contains the data and target values
    is_algorithm:   Instance selection algorithm, must be callable 
    aux_algoritm:   Auxiliar algorithm to be trained and used for the instance selection process, must be callable
    noise_perc:     Pecentange of instances with noise
    noise_magn:     Percentage of the amount noise on corrupted instances
    noise_type:     Defines the noise to be used, can be 'normal (gaussian)', 'uniform' or 'persize'
    split:          Number of folds used for cross validation
    filtr:         If it is equal to 1 apply instance selection, 0 not apply 
    rs:             Random state in cross validation

    return:
    A vector with RMSE and MAPE of target prediction, and F-SCORE, PRECISION AND RECALL OF noisy detection, and percentage of cases deleted
"""
def experiment_on_noise(dataset, is_algorithm, aux_algoritm, noise_perc, noise_magn=0.3, noise_type='uniform', filtr=1, split=10,  rs=30): 

    # KFold Cross Validation approach
    kf = KFold(n_splits=split, shuffle=True, random_state=rs)

    # Empty vectors to save predictions and real in each fold
    PREAGG=[]
    RECAGG=[]
    F1AGG=[]
    NUMAGG=[]
    predAgg= []
    YAgg= []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(dataset.x):
    
        # Split train-test
        X_train, X_test = dataset.x[train_index], dataset.x[test_index]
        Y_train, Y_test = dataset.y[train_index], dataset.y[test_index]
    
        # Add noise to target
        Y_train, idNoise = add_noise(target=Y_train, perc=noise_perc, method=noise_type, mag=noise_magn)    
        
        
        
        if filtr==1:
            # Apply instance selection algorithm
            idNoisePred = is_algorithm.evaluate(X_train,Y_train)
    
            # Delete Noise according to instance selection algorithmn
            X_train = X_train[idNoisePred != 1]
            Y_train = Y_train[idNoisePred != 1]
    
            # Evaluation Noise detection
            PRE = precision_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            REC = recall_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            F1 = f1_score(idNoise, idNoisePred, pos_label=1, zero_division=0)
            NUM = np.sum(idNoisePred)
            
            # Save evaluation Noise detection
            PREAGG = np.append(PREAGG, PRE)
            RECAGG = np.append(RECAGG, REC)
            NUMAGG = np.append(NUMAGG, NUM)
            F1AGG = np.append(F1AGG, F1)
        
        # Train the auxiliar algorithm
        aux_algoritm = aux_algoritm.fit(X_train, Y_train)
    
        # Generate prediction
        pred = aux_algoritm.predict(X_test)
    
        # Save real and prediction vectors of the  fold 
        predAgg = np.concatenate((predAgg, pred), axis=0)
        YAgg = np.concatenate((YAgg, Y_test), axis=0)

    
    # Evaluation
    RMSE = np.sqrt(mean_squared_error(YAgg, predAgg))
    MAPE = np.mean(np.abs(YAgg - predAgg) / (YAgg + 0.001))
    correlation_matrix = np.corrcoef(YAgg, predAgg)
    correlation_xy = correlation_matrix[0, 1]
    R2 = correlation_xy ** 2
    POR = np.sum(NUMAGG) / dataset.x.shape[0]
    if filtr == 1:
        output = np.array([RMSE, MAPE, R2, POR, np.mean(F1AGG), np.mean(RECAGG), np.mean(PREAGG)])
    else:
        output = np.array([RMSE, MAPE, R2,- 99, -99, -99, -99])
    
    return np.append(np.append(np.array([noise_perc, noise_magn, noise_type, filtr]) , output), [dataset.name, is_algorithm.name])


"""

    Function to evaluate a set of algorithms on a given dataset

    alg:            Dictionary containing the lambda function of the instance selection algorithms to evaluate
    data_path:      Path to the CSV file we'll work on
    noise_params:   
    
"""
def experiment_on_data(algorithms, data_path, noise_params):

    # Results container 
    results_df = pd.DataFrame(columns=["noise_perc", "noise_magn", "noise_type", "filter","RMSE","MAPE", "R2", "F1", "REC", "PRE", "POR", "DATA", "ALG"])

    # Get the data name
    name_data = data_path.split('/')[-1].split('.')[0]

    # Load dataset
    dataset = pd.read_csv(data_path, sep=",", header=None)
    # Normalize columns for x and y data
    dataset = preprocessing(dataset, dataset.shape[1] - 1)
    # Get the data and targets into a container to generate the meshgrid
    dataset = Dataset(name_data, dataset[0], dataset[1])

    # Get time stamps
    start_time = time.time()

    # Matrix of combined parameters for the experiments
    exp_params = np.array(np.meshgrid(  dataset,
                                        algorithms, 
                                        RandomForestRegressor(max_depth=9, random_state=0), 
                                        noise_params[0], # Number of added noisy instances (%)
                                        noise_params[1], # Percentage of noise added over the instance
                                        noise_params[2], # Noise type 
                                        noise_params[3]  # Filter 1 means yes
                                        )).T.reshape(-1, 7)

    # Add baseline tests

    
    # Start with the number of processors
    with Pool() as pool:
        # Run experiments in parallel and store results
        for noisy_result in pool.starmap(experiment_on_noise, exp_params):

            # Add the results to the dataframe
            results_df.loc[len(results_df)] = noisy_result

    exec_time = time.time() - start_time
    print("Finished experimenting on " + name_data + " dataset in " + str(int(exec_time // 60)) + ":" + str(int(exec_time % 60))) 

    # Reset the index in case addigng the rows caused indexing errors
    return results_df.reset_index(drop=True) 