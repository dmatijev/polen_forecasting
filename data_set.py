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
    def __init__(self, reads, k, nr_days, target, sim_label = ''):
        super().__init__()
        self.reads = reads[['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', f"{target if sim_label == '' else sim_label}"]]
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
    

class Dataset_SMSD(data.Dataset):
    def __init__(self, reads, k, nr_days, target, nr_sim):
        super().__init__()
        columns = ['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD']
        self.meteo = reads[columns]
        columns.extend([f'{i}-sim' for i in range(nr_sim)])
        self.reads = reads[columns]
        self.weights = reads['WGHT']
        self.labels = reads[target]
        self.k = k
        self.nr_days = nr_days
        self.target = target
        self.nr_sim = nr_sim


    def __getitem__(self, index):
        inputs = self.reads[index: index + self.k]
        meteo = self.meteo[index+self.k:index+self.k+self.nr_days]
        label = self.labels[index+self.k : index+self.k+self.nr_days]
        weights = self.weights[index+self.k : index+self.k+self.nr_days]
        inputs = torch.stack(tuple( \
                                   torch.tensor( inputs[ ['MNT','MKT', 'PAD', 'VLZ', 'MBV', 'RBD', f'{i}-sim'] ].to_numpy() , dtype = torch.float32 ) \
                                       for i in range(self.nr_sim) \
                                           ))
        
        return inputs, \
               torch.tensor(meteo.to_numpy(), dtype=torch.float32), \
               torch.tensor(label.to_numpy(), dtype = torch.float32), \
               torch.tensor(weights.to_numpy(), dtype = torch.float32)

    def __len__(self):
        return len(self.reads) - self.k - self.nr_days

