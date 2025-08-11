# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers.basic import BasicConv, Down, Up_Only, Outc,Up
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
    def __init__(self, dim,out_channels,patch_size=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, dim * patch_size * patch_size)
        self.output_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, H,W):
        B, N, C = x.shape
        x = self.proj(x)  # [B, N, patch*patch*C]
        x = x.view(B, int(H/self.patch_size), int(W/self.patch_size), C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        x = self.output_conv(x)
        return x



class TransformerUnet(nn.Module):
    
    def __init__(self,configs):
        super().__init__()
        self.configs = configs
        base_c=32
        patch_size=1
        dim=base_c*32
        depth=6
        #depth=12
        
        
        self.inc = BasicConv(1, base_c * 1, mid_channels=base_c)
        self.down1 = Down(base_c * 1, base_c * 2)     
        self.down2 = Down(base_c * 2, base_c * 4)     
        self.down3 = Down(base_c * 4, base_c * 8)     
        self.down4 = Down(base_c * 8, base_c * 16)     
        self.down5 = Down(base_c * 16, base_c * 32)     
        

        self.patch_embed = PatchEmbed(in_channels=base_c * 32, embed_dim=dim, patch_size=patch_size)
        self.encoder = TransformerEncoder(depth=depth, dim=dim)  #128 64 64 
        self.decoder = TransformerDecoder(dim=dim,out_channels=base_c * 32, patch_size=patch_size)
        
        
        self.up1 = Up(base_c * 32, base_c * 16)
        self.up2 = Up(base_c * 16, base_c * 8)
        self.up3 = Up(base_c * 8, base_c * 4)
        self.up4 = Up(base_c * 4, base_c * 2)
        self.up5 = Up(base_c * 2, base_c * 1)
        
        self.outc = Outc(base_c * 1, 1,mid_channels=base_c)
        

        
    def forward(self, x):
        
        if x.device.type!=self.configs.device:
            x=x.to(self.configs.device)
        B,C,H,W=x.shape 
        
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        y = self.down5(x4)
        
        y = self.patch_embed(y)
        y = self.encoder(y)
        y = self.decoder(y, H//32,W//32)
        
        y = self.up1(y, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.up5(y, x0)

        y=self.outc(y)
        

        return y
        
    

    
        
        