# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""
import torch
import torch.nn as nn

from network.layers.init_bias import Init_Bias
from network.layers.propagation_network import Propagation_Network
from network.layers.basic import Outc
from network.layers.denoise_network import Denoise_Network
from network.utils.tool import warp, make_grid

class UAE_Unet(nn.Module):
    
    def __init__(self,configs):
        super(UAE_Unet, self).__init__()
        self.configs = configs
        self.Denoise_Network=Denoise_Network(1,1,base_c=32)

    def forward(self, x):
        
        if x.device.type!=self.configs.device:
            x=x.to(self.configs.device)

        batch = x.shape[0]
        
        x=self.Denoise_Network(x)
        
        x = torch.clamp(x, min=0,max=255)
        
        return x
    

    



        
        