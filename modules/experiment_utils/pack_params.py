#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

import numpy as np

from .entities import Algorithm

"""
    Generates a list of Algorithm classes that execute one algorithm with 
    different parameters

    alg_name:   Algorithm name
    alg:        Algorithm function in this case only RegENN and DISKR functions
    ks:         Numpy array of k values
    alphas:     Numpy array of the posible alpha values

    returns     List of Algorithms with different parameters

"""
def pack_kalpha(alg_name, alg, ks, alphas):

    # Create the package
    packed_regenn = []

    # Generate a parameter mesh to optimize the class creation
    param_mesh = np.array(np.meshgrid(ks, alphas)).T.reshape(-1, 2)

    # Iterate over all the parameter combinations
    for k_value, alpha_value in param_mesh:

        # Build the algoritm name using the parameters
        name = "{}_k{}_alpha{}".format(alg_name, str(k_value), str(alpha_value))

        # Build the algorithm
        packed_regenn.append(Algorithm(name, alg, k=k_value, alpha=alpha_value))

    return packed_regenn



"""
    Generates a list of Algorithm classes that execute one algorithm with 
    different parameters

    alg_name:       Algorithm name
    alg:            Algorithm function in this case only RegBAG function
    ks:             Numpy array of k values
    alphas:         Numpy array of the posible alpha values
    num_bags:       Numpy array with the posible number of samples in bagging 
    sample_size:    Numpy array with the number of elements per sample. 

    returns     List of Algorithms with different parameters

"""
def pack_RegBAG(alg_name, alg, ks, alphas, num_bags, sample_size):

    # Create the package
    packed_regbag = []

    # Generate a parameter mesh to optimize the class creation
    param_mesh = np.array(np.meshgrid(ks, alphas, num_bags, sample_size)).T.reshape(-1, 4)

    # Iterate over all the parameter combinations
    for k_value, alpha_value, bags_value, size_value in param_mesh:

        # Build the algoritm name using the parameters
        name = "{}_k{}_alpha{}_numBags{}_sampsize{}".format(alg_name, 
                                                            str(k_value), 
                                                            str(alpha_value),
                                                            str(bags_value), 
                                                            str(size_value))

        # Build the algorithm
        packed_regbag.append(Algorithm( name, alg, 
                                        k=k_value, 
                                        alpha=alpha_value, 
                                        numBag=bags_value,
                                        samSize=size_value))

    return packed_regbag


"""
    Generates a list of Algorithm classes that execute one algorithm with 
    different parameters

    alg_name:       Algorithm name
    alg:            Algorithm function in this case only DiscENN function
    ks:             Numpy array of k values
    strat:          Numpy array with posible strategies for the KBins Discretizer
    bins:           Numpy array with posible bin sizes for the KBins Discretizer

    returns     List of Algorithms with different parameters

"""
def pack_DiscENN(alg_name, alg, ks, strat, bins):

    # Create the package
    packed_discenn = []

    # Generate a parameter mesh to optimize the class creation
    param_mesh = np.array(np.meshgrid(ks, strat, bins)).T.reshape(-1, 3)

    # Iterate over all the parameter combinations
    for k_value, strat_value, bins_value in param_mesh:

        # Build the algoritm name using the parameters
        name = "{}_k{}_bins{}".format(  alg_name, 
                                                str(k_value), 
                                                #str(strat_value),
                                                str(bins_value))

        # Build the algorithm
        packed_discenn.append(Algorithm(name, alg, 
                                        k=k_value, 
                                        #strat=strat_value, 
                                        bins=bins_value))

    return packed_discenn


"""
    Generates a list of Algorithm classes that executes DROP2RE algorithm with 
    different parameters

    alg_name:       Algorithm name
    alg:            Algorithm function in this case only DROP2RE function
    ks:             Numpy array of k values

    returns     List of Algorithms with different parameters

"""
def pack_DROP2RE(alg_name, alg, ks):

    # Create the package
    packed_drop2re = []

    # Iterate over all the parameter combinations
    for k_value in ks:

        # Build the algoritm name using the parameters
        name = "{}_k{}".format( alg_name, 
                                str(k_value))

        # Build the algorithm
        packed_drop2re.append(Algorithm(name, alg, 
                                        k=k_value))

    return packed_drop2re


def pack_algorithms(algorithms, k, regs_alpha, diskr_alpha, num_bags, sample_size, strategy, num_bins):

    # Create the package
    packed_algorithms = []

    # Iterate over all the algorithms so we can choose wich parameters 
    # use to build the class
    for alg_name, alg in algorithms.items():
        
        # Build the RegENN algorithms the same way since they have the same parameters
        if("RegENN" in alg_name):
            packed_algorithms += pack_kalpha(alg_name, alg, k, regs_alpha)

        # Build the RegENN algorithms the same way since they have the same parameters
        elif("RegENN2" in alg_name):
            packed_algorithms += pack_kalpha(alg_name, alg, k, regs_alpha)
       
	# Build the RegENN algorithms the same way since they have the same parameters
        elif("RegENN3" in alg_name):
            packed_algorithms += pack_kalpha(alg_name, alg, k, regs_alpha)

        # Build the RegBAG algorithm
        elif("RegBAG" in alg_name):
            packed_algorithms += pack_RegBAG(alg_name, alg, k, regs_alpha, num_bags, sample_size)

        # Build the DiscENN algorithm
        elif("DiscENN" in alg_name):
            packed_algorithms += pack_DiscENN(alg_name, alg, k, strategy, num_bins)
        
        # Build the DISKR algorithm
        elif("DISKR" in alg_name):
            packed_algorithms += pack_kalpha(alg_name, alg, k, diskr_alpha)

        # Build the DROP2RE algorithm
        elif("DROP2RE" in alg_name):
            packed_algorithms += pack_DROP2RE(alg_name, alg, k)

        # Catch errors in names
        else:
            sys.exit("Invalid algoritm name detected")


    return packed_algorithms

