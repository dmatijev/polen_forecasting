# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:36:55 2021

@author: dmatijev
"""
import torch
import torch.utils.data as data



def weighted_mse_loss(reduction):
    """
    loss function that takes inputs, labels and weights. 
    Be sure that inputs.shape == labels.shape == weights.shape, 
    outherwise, strange things can happen :-)
    """
    if reduction == 'mean':
        return lambda _input, _label, _weights: (_weights * (_input - _label) ** 2).mean()
    elif reduction == 'sum':
        return lambda _input, _label, _weights: (_weights * (_input - _label) ** 2).sum() 
    else:
        raise Exception("reduction flag should be set to either 'mean' or 'sum' ")

class Dataset(data.Dataset):
    def __init__(self, reads, k, nr_days, target, nr_sim = ''):
        super().__init__()
        #self.reads = reads.drop(['WGHT'], axis=1)
        self.reads = reads[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', f"{target if nr_sim == '' else nr_sim}"]]
        self.meteo = reads[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD']]
        self.weights = reads['WGHT']
        self.labels = reads[target]
        self.k = k
        self.nr_days = nr_days
        self.target = target


    def __getitem__(self, index):
        inputs = self.reads[index: index + self.k]
        meteo = self.meteo[index+self.k:index+self.k+self.nr_days]
        label = self.labels[index+self.k : index+self.k+self.nr_days]
        weights = self.weights[index+self.k : index+self.k+self.nr_days]
   
        return torch.tensor(inputs.to_numpy(), dtype = torch.float32), \
               torch.tensor(meteo.to_numpy(), dtype=torch.float32), \
               torch.tensor(label.to_numpy(), dtype = torch.float32), \
               torch.tensor(weights.to_numpy(), dtype = torch.float32)

    def __len__(self):
        return len(self.reads) - self.k - self.nr_days
    
