# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:44:43 2023

@author: Administrator
"""

import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_grid(B,C,H,W,device):

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    grid[:,0, :, :] = 2.0 * grid[:,0, :, :] / max(H - 1, 1) - 1.0
    grid[:,1, :, :] = 2.0 * grid[:,1, :, :] / max(W - 1, 1) - 1.0
    grid = grid.to(device)
    
    return grid

def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):

    B, C, H, W = input.size()
    vgrid = grid + flow

    vgrid[:, 0, :, :] = 3.0 * vgrid[:, 0, :, :].clone()  - 1.5
    vgrid[:, 1, :, :] = 3.0 * vgrid[:, 1, :, :].clone()  - 1.5
    vgrid = vgrid.permute(0, 2, 3, 1)

    output = torch.nn.functional.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output

def mse(target,folder_list,num=32,times=[]):

   
    if times==[]:
        times=[1 for i in range(len(folder_list))]
    
    
    error=[[0 for _ in range(num)] for i in range(len(folder_list))]
    error_min=[[np.inf for _ in range(num)] for i in range(len(folder_list))]
    error_max=[[0 for _ in range(num)] for i in range(len(folder_list))]
    
    temp=os.listdir(folder_list[0])
    
    for i in temp:
        
        path0=target+"/"+i
        
        accumulate=[0 for _ in range(len(folder_list))]
        
        for j in range(num):
            
            img0 = Image.open(f"{path0}/{str(j).zfill(8)}.png") 
            
            img0=np.array(img0)
            
            img0=img0/255
            
            for k,path_plot in enumerate(folder_list):
            
                path1=path_plot+"/"+i
                
                img1 = Image.open(f"{path1}/{str(j).zfill(8)}.png")
            
                img1=np.array(img1)
                img1=img1//times[k]
                img1=img1/255
                
                #a=np.mean((img0-img1)**2)/np.mean(img0)+accumulate[k]
                a=np.mean((img0-img1)**2)+accumulate[k]
                
                accumulate[k]=a
                
                error_min[k][j]=min(accumulate[k],error_min[k][j])
                error_max[k][j]=max(accumulate[k],error_max[k][j])
                
                error[k][j]+=a/len(temp)
    

    return error,error_max,error_min


