# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""

    

import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, configs):
        super(FC, self).__init__()
        
        
        input_dim=10
        
        output_dim=1
        
        hidden_dims=[10,10]
        
        
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
