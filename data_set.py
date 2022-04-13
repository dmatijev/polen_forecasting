# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:36:55 2021

@author: dmatijev
"""
import torch
import torch.utils.data as data



def weighted_mse_loss(weights, reduction):
    if reduction == 'mean':
        return lambda _input, _label: (weights * (_input - _label) ** 2).mean()
    elif reduction == 'sum':
        return lambda _input, _label: (weights * (_input - _label) ** 2).sum()
    else:
        raise Exception("reduction flag should be set to either 'mean' or 'sum' ")

class Dataset(data.Dataset):
    def __init__(self, reads, k, nr_days, target):
        super().__init__()
        self.reads = reads
        self.k = k
        self.nr_days = nr_days
        self.target = target


    def __getitem__(self, index):
        # implement taking k-plets at index position. Label is the polen 
        # forcast at index + (k+1) position
        # k_plet format is seq_len times input_dim
        
        k_plet = self.reads[index: index + self.k]
        l_plet = self.reads[index+self.k:index+self.k+self.nr_days]
        label = self.reads[self.target][index+self.k : index+self.k+self.nr_days]

        return torch.tensor(k_plet.to_numpy(), dtype = torch.float32), torch.tensor(l_plet.to_numpy(), dtype=torch.float32), torch.tensor(label.to_numpy(), dtype = torch.float32)

    def __len__(self):
        return len(self.reads) - self.k - self.nr_days
    
