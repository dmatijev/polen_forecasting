# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:40:36 2021

@author: Administrator
"""

import pandas as pd
from math import log
import numpy as np

#data = pd.read_csv('real_for_all_podaci.csv')

def train_test_split(data, target, locations = ['NS'], test_years = [2015,2016]):
    data = data[data['LOK'].isin(locations)]
    test_dataset = data[data['GOD'].isin(test_years)]
    train_dataset = data[~(data['GOD'].isin(test_years))]
    return train_dataset, test_dataset

def standardize(dataset, d_mean, d_std):
    return (dataset - d_mean)/d_std

def normalize_stari(dataset, d_min, d_max):
    return (dataset - d_min)/(d_max - d_min)



def normalize(dataset, d_mean):
    print(d_mean)
    return dataset/d_mean

def logarithm(dataset, train_dataset):
    # normalizirati
    dataset = normalize(dataset, train_dataset)
    # logaritmirati
    mins = dataset.min()
    columns = list(dataset.columns)
    for i, col in enumerate(columns):
        if mins[i] < 1e-9:
            dataset[col] = np.log(dataset[col] + 1)
        else:
            dataset[col] = np.log(dataset[col])
    return dataset

def takeSeasons(dataset, target):
    if target == 'PRAM':
        dataset = dataset[dataset['MSC'].isin([7,8,9,10])]
    elif target == 'PRBR':
        dataset = dataset[dataset['MSC'].isin([3,4,5])]
    else:
        dataset = dataset[dataset['MSC'].isin([4,5,6,7,8,9,10])]
    
    return dataset
    

# TRAIN SET: 2000-2013
# VALID SET: 2014-2015
#  TEST SET: 2016-2017
# TARGET: PRAM (AMBROZIJA)



def load_data(file_name, preproc = 'lognormalize' ):
    data = pd.read_csv(file_name)
    data = data[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM', 'GOD', 'LOK', 'MSC']]
    # pomaknuti temperaturu
    data[['MKT', 'MNT']] = (data[['MKT', 'MNT']] + 40)/90.0
    trainvalid_dataset, test_dataset = train_test_split(data, 'PRAM', locations = ['NS'], test_years=[2015,2016])
    train_dataset, valid_dataset = train_test_split(trainvalid_dataset, 'PRAM', locations = ['NS'], test_years = [2013,2014])


    #train_dataset = takeSeasons(train_dataset, 'PRAM')
    #valid_dataset = takeSeasons(valid_dataset, 'PRAM')
    #test_dataset = takeSeasons(test_dataset, 'PRAM')
    
    #train_pram = train_dataset['PRAM']
    #valid_pram = valid_dataset['PRAM']
    #test_pram = test_dataset['PRAM']
    train_dataset_mnt = train_dataset['MSC']
    train_dataset = train_dataset[['MNT', 'MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]

    #train_dataset = train_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD']]
    valid_dataset_mnt = valid_dataset['MSC']
    valid_dataset = valid_dataset[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]

    #valid_dataset = valid_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD']]
    test_dataset_mnt = test_dataset['MSC']
    test_dataset = test_dataset[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]

    #test_dataset = test_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD']]


    train_min = train_dataset.min()
    train_max = train_dataset.max()
    train_mean = train_dataset.mean()
    train_std = train_dataset.std()
    if preproc == 'lognormalize':
        train_dataset = logarithm(train_dataset, train_mean)
        valid_dataset = logarithm(valid_dataset, train_mean)
        test_dataset = logarithm(test_dataset, train_mean)
    elif preproc == 'standardize':
        train_dataset = standardize(train_dataset, train_mean, train_std)
        valid_dataset = standardize(valid_dataset, train_mean, train_std)
        test_dataset = standardize(test_dataset, train_mean, train_std)
    else:
        train_dataset = normalize_stari(train_dataset, train_min, train_max)
        valid_dataset = normalize_stari(valid_dataset, train_min, train_max)
        test_dataset = normalize_stari(test_dataset, train_min, train_max)

    #train_dataset['PRAM'] = train_pram
    #valid_dataset['PRAM'] = valid_pram
    #test_dataset['PRAM'] = test_pram
    train_dataset['MSC'] = train_dataset_mnt
    valid_dataset['MSC'] = valid_dataset_mnt
    test_dataset['MSC'] = test_dataset_mnt

    train_dataset = takeSeasons(train_dataset, 'PRAM')
    valid_dataset = takeSeasons(valid_dataset, 'PRAM')
    test_dataset = takeSeasons(test_dataset, 'PRAM')

    train_dataset = train_dataset.reset_index()
    valid_dataset = valid_dataset.reset_index()
    test_dataset = test_dataset.reset_index()

    train_dataset = train_dataset[['MNT', 'MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
    valid_dataset = valid_dataset[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
    test_dataset = test_dataset[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]

    
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = load_data('real_for_all_podaci.csv')