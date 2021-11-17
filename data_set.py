# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:36:55 2021

@author: dmatijev
"""
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, reads, k):
        super().__init__()
        self.reads = reads
        self.k = k

        

    def __getitem__(self, index):
        # implement taking k-plets at index position. Label is the polen 
        # forcast at index + (k+1) position
        # k_plet format is seq_len times input_dim
        
        k_plet = self.reads[index: index + self.k]
        label = self.reads['PRAM'][index + self.k + 1]

        return torch.tensor(k_plet.to_numpy(), dtype = torch.float32), torch.tensor(label, dtype = torch.float32)

    def __len__(self):
        return len(self.reads) - self.k - 1
    
