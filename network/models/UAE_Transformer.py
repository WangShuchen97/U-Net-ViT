# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers.basic import BasicConv, Down, Up_Only, Outc
from network.layers.denoise_network import Denoise_Network
    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,dropout=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2=nn.GELU()
        self.fc3=nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_features, out_features)
        self.fc5=nn.Dropout(dropout)
    
    def forward(self, x):
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        x=self.fc5(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim,patch_size=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, dim * patch_size * patch_size)
        
        self.up1 = Up_Only(dim, dim//2)
        self.up2 = Up_Only(dim//2, dim//4)
        self.up3 = Up_Only(dim//4, dim//8)
        self.up4 = Up_Only(dim//8, dim//16)
        self.outc = Outc(dim//16, 1,mid_channels=dim//32)
        
    
    def forward(self, x, H,W):
        B, N, C = x.shape
        x = x.view(B, int(H/self.patch_size), int(W/self.patch_size), C, 1, 1)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, int(H/self.patch_size), int(W/self.patch_size))
        
        x=self.up1(x)
        x=self.up2(x)
        x=self.up3(x)
        x=self.up4(x)
        x=self.outc(x)
      
        return x

    
class Transformer(nn.Module):
    
    def __init__(self,configs):
        super().__init__()
        self.configs = configs
        patch_size=16
        dim=512
        depth=6
        #depth=12

        self.patch_embed = PatchEmbed(in_channels=1, embed_dim=dim, patch_size=patch_size)
        self.encoder = TransformerEncoder(depth=depth, dim=dim)  #128 64 64 
        self.decoder = TransformerDecoder(dim=dim, patch_size=patch_size)
        
    def forward(self, x):
        
        if x.device.type!=self.configs.device:
            x=x.to(self.configs.device)
        B,C,H,W=x.shape 

        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.decoder(x, H,W)
        

        return x
        
    

    
        
        