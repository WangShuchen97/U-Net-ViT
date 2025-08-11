# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:09:38 2023

@author: Administrator
"""

from torch.utils.data import DataLoader
import os
import torch
from sklearn.model_selection import train_test_split
import random

#==============================================================

from network.data_provider import Img_Img,txt_txt
datasets_map = {
    'Img_Img': Img_Img.InputHandle,
    'txt_txt': txt_txt.InputHandle
}
#==============================================================

def data_provider(configs,transform=None,mode=None):
    


    if configs.data_provider not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)
    datasets=datasets_map[configs.data_provider]
    
    if mode is None:
        mode=configs.mode
    if mode=='train':
        is_training=True
    else:
        is_training=False
        
    case_list=[]
    target_list=[]
    
    num=0        
    for name in os.listdir(configs.dataset_input):
        case_list.append(configs.dataset_input+'/' + name)
        target_list.append(configs.dataset_output+'/' + name)
        num+=1
        if num >=configs.maximum_sample_size:
            break
    
    if is_training:
        train_data, test_data, train_targets, test_targets = train_test_split(case_list, target_list, test_size=configs.test_ratio, random_state=configs.seed)
        if configs.is_val:
            train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_targets, test_size=configs.val_ratio, random_state=configs.seed)
    else:
        random.seed(configs.seed)
        random_list = list(zip(case_list, target_list))
        random.shuffle(random_list) 
        test_data, test_targets = zip(*random_list)  
    
    
    train_input_handle=None
    val_input_handle=None
    test_input_handle=None
    
    test_input_handle = datasets(test_data,test_targets,configs,transform)
    test_input_handle = DataLoader(test_input_handle,
                                   batch_size=configs.batch_size_test,
                                   shuffle=False,
                                   num_workers=configs.cpu_worker,
                                   drop_last=False)
    if is_training:
        train_input_handle = datasets(train_data,train_targets,configs,transform)
        if configs.is_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(train_input_handle, num_replicas=configs.world_size, rank=configs.rank)
            train_input_handle = DataLoader(train_input_handle,
                                           batch_size=configs.batch_size,
                                           num_workers=configs.cpu_worker,
                                           drop_last=True,
                                           sampler=sampler,
                                           pin_memory=True)
        else:    
            train_input_handle = DataLoader(train_input_handle,
                                           batch_size=configs.batch_size,
                                           shuffle=True,
                                           num_workers=configs.cpu_worker,
                                           drop_last=True)
        if configs.is_val:
            val_input_handle = datasets(val_data,val_targets,configs,transform)
            val_input_handle = DataLoader(val_input_handle,
                                           batch_size=configs.batch_size_val,
                                           shuffle=False,
                                           num_workers=configs.cpu_worker,
                                           drop_last=False)
            
    return train_input_handle,val_input_handle,test_input_handle

