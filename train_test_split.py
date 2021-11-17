# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:40:36 2021

@author: Administrator
"""

import pandas as pd

#data = pd.read_csv('real_for_all_podaci.csv')

def train_test_split(data, target, locations = ['NS'], test_years = [2015,2016]):
    data = data[data['LOK'].isin(locations)]
    test_dataset = data[data['GOD'].isin(test_years)]
    train_dataset = data[~(data['GOD'].isin(test_years))]
    return train_dataset, test_dataset

def standardize(dataset):
    return (dataset - dataset.mean())/dataset.std()

def normalize(dataset):
    return (dataset - dataset.min())/(dataset.max() - dataset.min())


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


def load_data(file_name):
    data = pd.read_csv(file_name)
    data = data[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM', 'GOD', 'LOK', 'MSC']]
    trainvalid_dataset, test_dataset = train_test_split(data, 'PRAM', locations = ['NS'], test_years=[2016,2017])
    train_dataset, valid_dataset = train_test_split(trainvalid_dataset, 'PRAM', locations = ['NS'], test_years = [2014,2015])
    
    train_dataset = takeSeasons(train_dataset, 'PRAM')
    valid_dataset = takeSeasons(valid_dataset, 'PRAM')
    test_dataset = takeSeasons(test_dataset, 'PRAM')
    
    
    
    
    train_dataset = train_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
    valid_dataset = valid_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
    test_dataset = test_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
    
    train_dataset = normalize(train_dataset)
    valid_dataset = normalize(valid_dataset)
    test_dataset = normalize(test_dataset)
    
    
    return train_dataset, valid_dataset, test_dataset



train_dataset, valid_dataset, test_dataset = load_data('real_for_all_podaci.csv')