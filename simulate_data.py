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
    parser.add_argument('--ws', type=str, required=True)
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

def weight_invers_std(std):
    return 1/(2*std)

def weight_jump(y, y_prev, min_ch, max_ch):
    return 1 + 9*np.abs(((y-y_prev) -min_ch)/(max_ch - min_ch))

if __name__ == "__main__":
    args = get_parser().parse_args()
    data_file = args.file_name 
    R = args.nr_datasets
    target = args.target
    ws = args.ws
    
    data = pd.read_csv(data_file)
    data_rows = data.shape[0]
    data_cols = data.shape[1]
    
    
    for i in range(R):
        data[f'{i}-sim'] = 0
    data['WGHT'] = 1 #where stDev is 0, weight remains 1
        
    print ("Simulating datasets... ")
    max_ch = -np.inf
    min_ch = np.inf
    if ws == 'jump_weights':
        for i in range(1,data_rows):
            y_prev = data.iloc[i-1, data.columns == target].item()
            y = data.iloc[i, data.columns == target].item()
            chg = np.abs(y - y_prev)
            if chg > max_ch:
                max_ch = chg
            if chg < min_ch:
                min_ch = chg
         
    for i in range(1,data_rows):
        targetValue = data.iloc[i, data.columns == target].item()
        if i == 0:
            targetValuePrev = 0
        else:
            targetValuePrev = data.iloc[i-1, data.columns == target].item()
        stDev = (targetValue*getDeviation(targetValue))/3 #normal distribution, the values less than three standard deviations account for 99.73%
        data.iloc[i, data_cols:data_cols + R] = [targetValue]*R + np.random.normal(0, 2*stDev, R)
        if ws == 'invers_std':
            if stDev != 0:
                data.iloc[i, data_cols + R] = weight_invers_std(stDev) #data_cols + R is 'WGHT' column
        elif ws == 'jump_weights':
            data.iloc[i, data_cols + R] = weight_jump(targetValue,targetValuePrev,min_ch,max_ch)
        #else:
        #    data.iloc[i, data_cols + R] = 100 #stdev is=0 weight should be large to penalize differences
        
    
    print(f"saving new datsets to sim-{R}-{target}-{ws}-{data_file} ", end ="")
    data.to_csv(f'sim-{R}-{target}-{ws}-{data_file}', index=False)
    print("Enjoy!")