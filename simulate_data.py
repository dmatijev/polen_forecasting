# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:54:52 2022

@author: dmatijev
"""

import numpy as np
import pandas as pd
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--nr_datasets', type=int, required=True)
    return parser



#data_file = 'real_for_all_podaci.csv'

#R = 10 # number of datasets

if __name__ == "__main__":
    args = get_parser().parse_args()
    data_file = args.file_name 
    R = args.nr_datasets
    
    data = pd.read_csv(data_file)
    data_rows = data.shape[0]
    data_cols = data.shape[1]
    
    mu = np.zeros(data_rows)
    sigma = np.random.rand(data_rows) # simulate variances... for now... 
    
    for i in range(R):
        data[f'{i}-sim'] = 0
        
    for i in range(data_rows):
        data.iloc[i, data_cols:data_cols + R] = np.random.normal(mu[0], sigma[0], R)
        
    
    print(f"saving new datset to sim-{R}-{data_file}  ", end ="")
    data.to_csv(f'sim-{R}-{data_file}', index=False)
    print("Done!")
    
    #mu, sigma = num0, 0.1
    #s = np.random.normal(mu, sigma, R)