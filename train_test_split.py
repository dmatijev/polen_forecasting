# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:40:36 2021

@author: Administrator
"""

import pandas as pd
from math import log
import numpy as np


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
    #print(d_mean)
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


def load_data(file_name, test_years=[2015, 2016], valid_years=[2013,2014], exclude_years=[2017, 2018, 2019, 2020, 2021], preproc = 'lognormalize',target='PRAM', nr_sim = 0, use_weights = False):
    data = pd.read_csv(file_name)
    data=data[~data['GOD'].isin(exclude_years)] #remove years not needed for training, validation and testing
    listOfColumns=['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', target,  'GOD', 'LOK', 'MSC', 'WGHT']
    if nr_sim > 0:
        for nd in range(nr_sim):
            listOfColumns.append(f"{nd}-sim")
    data = data[listOfColumns]
    
    # move temperature so that values are greater than 0 (because of the logarithm)
    data[['MKT', 'MNT']] = (data[['MKT', 'MNT']] + 40)/90.0
    trainvalid_dataset, test_dataset = train_test_split(data, target, locations = ['NS'], test_years=test_years)
    train_dataset, valid_dataset = train_test_split(trainvalid_dataset, target, locations = ['NS'], test_years = valid_years)

    listOfColumns = [el for el in listOfColumns if el not in ['GOD', 'LOK', 'MSC', 'WGHT']]
                
    train_dataset_wght = train_dataset['WGHT']
    valid_dataset_wght = valid_dataset['WGHT']
    test_dataset_wght = test_dataset['WGHT']
    
    train_dataset_mnt = train_dataset['MSC']
    train_dataset = train_dataset[listOfColumns]

    valid_dataset_mnt = valid_dataset['MSC']
    valid_dataset = valid_dataset[listOfColumns]

    test_dataset_mnt = test_dataset['MSC']
    test_dataset = test_dataset[listOfColumns]
    
    
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

    train_dataset['MSC'] = train_dataset_mnt
    valid_dataset['MSC'] = valid_dataset_mnt
    test_dataset['MSC'] = test_dataset_mnt
    train_dataset['WGHT'] = train_dataset_wght
    valid_dataset['WGHT'] = valid_dataset_wght
    test_dataset['WGHT'] = test_dataset_wght

    
    train_dataset = takeSeasons(train_dataset, target)
    valid_dataset = takeSeasons(valid_dataset, target)
    test_dataset = takeSeasons(test_dataset, target)

    train_dataset = train_dataset.reset_index()
    valid_dataset = valid_dataset.reset_index()
    test_dataset = test_dataset.reset_index()
    
    listOfColumns.append('WGHT')
    train_dataset = train_dataset[listOfColumns]
    valid_dataset = valid_dataset[listOfColumns]
    test_dataset = test_dataset[listOfColumns]


    if not use_weights: # all weights are set to 1
        train_dataset['WGHT'] = pd.Series(np.ones(train_dataset.shape[0]))
        valid_dataset['WGHT'] = pd.Series(np.ones(valid_dataset.shape[0]))
        test_dataset['WGHT'] = pd.Series(np.ones(test_dataset.shape[0]))
    
 
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    TARGET = 'PRAM'
    train_dataset, valid_dataset, test_dataset = load_data(f'sim-30-{TARGET}-jump_weights-real_for_all_NS_2000_2021.csv', [2015,2016], [2013_2014], [2017, 2018,2019,2020,2021], 'lognormalize', TARGET,  2, True)