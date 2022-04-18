# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:54:52 2022

@author: dmatijev
"""

import numpy as np
import pandas as pd
import argparse
import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--nr_datasets', type=int, required=True)
    parser.add_argument('--target', type=str, required=True)
    return parser

def getDeviation(x):
    if 10 < x <= 30:
        return 0.3
    elif 30 < x <=300:
        return 0.2
    elif 300 < x:
        return 0.1
    else:
        return 0


#data_file = 'real_for_all_podaci_novo.csv'

#R = 10 # number of datasets

if __name__ == "__main__":
    args = get_parser().parse_args()
    data_file = args.file_name 
    R = args.nr_datasets
    target = args.target
    
    data = pd.read_csv(data_file)
    data_rows = data.shape[0]
    data_cols = data.shape[1]
    
    #mu = np.zeros(data_rows)
    #sigma = np.random.rand(data_rows) # simulate variances... for now...
    
    for i in range(R):
        data[f'{i}-sim'] = 0
        
    print ("Simulating datasets... ")
    for i in tqdm.tqdm(range(data_rows)):
        targetValue = data.iloc[i, data.columns == target].item()
        data.iloc[i, data_cols:data_cols + R] = [targetValue]*R + np.random.normal(0, (targetValue*getDeviation(targetValue))/3, R) #normal distribution, the values less than three standard deviations account for 99.73%
        
    
    print(f"saving new datsets to sim-{R}-{target}-{data_file}  ", end ="")
    data.to_csv(f'sim-{R}-{target}-{data_file}', index=False)
    print("Enjoy!")
    
    #mu, sigma = num0, 0.1
    #s = np.random.normal(mu, sigma, R)