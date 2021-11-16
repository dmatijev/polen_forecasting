# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:40:36 2021

@author: Administrator
"""

import pandas as pd

data = pd.read_csv('real_for_all_podaci.csv')

def train_test_split(data, target, locations = ['NS'], test_seasons = [2015,2016]):  
    data = data[data['LOK'].isin(locations)]
    test_dataset = data[data['GOD'].isin(test_seasons)]
    #test_dataset = test_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD'] + [target]]
    train_dataset = data[~(data['GOD'].isin(test_seasons))]
    #train_dataset = train_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD'] + [target]]
    
    return train_dataset, test_dataset



# TRAIN SET: 2000-2013
# VALID SET: 2014-2015
#  TEST SET: 2016-2017
# TARGET: PRAM (AMBROZIJA)

trainvalid_dataset, test_dataset = train_test_split(data, 'PRAM', locations = ['NS'], test_seasons=[2016,2017])
train_dataset, valid_dataset = train_test_split(trainvalid_dataset, 'PRAM', locations = ['NS'], test_seasons = [2014,2015])

train_dataset = train_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
valid_dataset = valid_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]
test_dataset = test_dataset[['MNT', 'PAD', 'VLZ', 'MBV', 'RBD', 'PRAM']]