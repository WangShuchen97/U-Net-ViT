# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:43:52 2023

@author: Administrator
"""

import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class InputHandle(Dataset):
    def __init__(self,data,targets=None,configs=None,transform=None):
        self.data=data
        self.targets=targets
        self.configs=configs
        self.transform = transform
        self.input_data_type='float32'
        self.output_data_type='float32'
        if not (self.configs is None):
            random.seed(self.configs.seed)  
            self.input_mean=self.configs.input_mean
            self.input_std=self.configs.input_std
            if self.configs.mode=='train':
                self.p_RandomHorizontalFlip=self.configs.p_RandomHorizontalFlip
                self.p_RandomVerticalFlip=self.configs.p_RandomVerticalFlip
                self.p_RandomRotate=self.configs.p_RandomRotate
            else:
                self.p_RandomHorizontalFlip=0
                self.p_RandomVerticalFlip=0
                self.p_RandomRotate=0
     
    def random_transform_together(self):
        
        if self.p_RandomHorizontalFlip>random.random():
            pp_RandomHorizontalFlip=1
        else:
            pp_RandomHorizontalFlip=0
        if self.p_RandomVerticalFlip>random.random():
            pp_RandomVerticalFlip=1
        else:
            pp_RandomVerticalFlip=0
                   
        temp=[transforms.RandomHorizontalFlip(pp_RandomHorizontalFlip),                                                                                                
              transforms.RandomVerticalFlip(pp_RandomVerticalFlip) ]
        if self.p_RandomRotate>random.random():
            pp_RandomRotate=random.randint(0,360)
            RandomRotate = lambda x: TF.rotate(x, pp_RandomRotate)
            temp.append(transforms.Lambda(RandomRotate))
        transform = transforms.Compose(temp)            
        return transform
    
    
    def load(self, index):
        data_name = self.targets[index]
    

        # 读取输入数据
        with open(self.data[index], 'r') as f:
            line = f.read().strip()
        sample = [float(x) for x in line.split(',')]
        sample = np.array(sample, dtype=self.input_data_type)
        sample = torch.tensor(sample)

    
        # 读取目标数据
        with open(self.targets[index], 'r') as f:
            line = f.read().strip()
        target = [float(x) for x in line.split(',')]
        target = np.array(target, dtype=self.output_data_type)
        target = torch.tensor(target)


        target = target * self.configs.output_times
    
        return sample, target, data_name

    # def load(self, index):

        
    #     data_name=self.targets[index]

    #     transform_together=self.random_transform_together()
    #     sample = cv2.imread(self.data[index], 2)
    #     sample=sample.astype(self.input_data_type)
    #     sample=transforms.ToTensor()(sample)
        
    #     if len(self.input_mean)>0 and len(self.input_mean)==len(self.input_std):
    #         sample=transforms.Normalize(self.input_mean,self.input_std)(sample)
        
    #     if not (self.transform is None):
    #         sample = self.transform(sample)
    #     sample = transform_together(sample)
        
        

    #     target=cv2.imread(self.targets[index], 2)
    #     target=target.astype(self.output_data_type)
    #     target=transforms.ToTensor()(target)
    #     if not (self.transform is None):
    #         target = self.transform(target)
    #     target = transform_together(target)
    #     target=target*self.configs.output_times
    #     return sample,target,data_name

    
    def __getitem__(self, index):
        sample, target, data_name=self.load(index)
        return sample, target, data_name

    def __len__(self):
        return len(self.data)
        
