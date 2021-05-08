#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
from torch import nn
from torch.nn import functional as F
from .model_util import torch_expm, construct_affine

sys.path.append("..")
from data.dataset import PILRandomGaussianBlur, get_color_distortion, KorniaAugmentationPipeline

def get_transformer(name):
    transformers = {'affine_RNN': ST_Affine_RNN,
                    'affinediff': ST_AffineDiff,
                    }
    assert (name in transformers), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in transformers.keys()])
    return transformers[name]

def expm(theta): 
    n_theta = theta.shape[0] 
    zero_row = torch.zeros(n_theta, 1, 3, dtype=theta.dtype, device=theta.device) 
    theta = torch.cat([theta, zero_row], dim=1) 
    theta = torch_expm(theta) 
    theta = theta[:,:2,:] 
    return theta

class ST_Affine_RNN(nn.Module):
    def __init__(self, input_shape, hidden_shape, num_steps=2):
        super(ST_Affine_RNN, self).__init__()
        assert len(input_shape)==3, 'input shape should have 3 dims: (C, H, W)'
        
        self.input_shape = input_shape # should be without the batch dimension
        self.hidden_shape = hidden_shape # size of the hidden features
        self.num_steps = num_steps # number of steps for the sequential model (i.e. number of views to produce)
        self.conv_channels = 16 # number of channels in the extracted features 
        
        # Feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], self.conv_channels, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2), # halves the spatial dims
            nn.ReLU(True),
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2), # halves the spatial dims
            nn.ReLU(True)
        )
        
        # feature extractor reduces each spatial dimension by a factor of 4
        out_h, out_w = self.input_shape[1] // 4 , self.input_shape[2] // 4 
        self.rnn_input_shape = self.conv_channels * out_h * out_w
        # Sequential model (Gated Recurrent Unit) for producing a sequence of views
        self.RNN = nn.GRUCell(self.rnn_input_shape, self.hidden_shape)
        
        # Regressor for 3 x 2 affine matrix
        self.fc_theta = nn.Sequential(
            nn.Linear(self.hidden_shape , self.hidden_shape//2),
            nn.ReLU(True),
            nn.Linear(self.hidden_shape//2, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_theta[2].weight.data.zero_()
        self.fc_theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Since the cropping is handled by STN, we need to move the rest
        # of the augmentations from the dataloader to here, and make them differentiable. 
        self.transform = KorniaAugmentationPipeline()


    def forward(self, x, h_0):
        # ==== extract features ==== #
        feats = self.conv(x)
        feats = feats.view(-1, self.rnn_input_shape)

        # ==== produce <num_steps> many views ==== #
        views = []
        thetas = []
        curr_h = h_0
        for step in range(self.num_steps):
            # regress the affine matrix
            curr_h = self.RNN(feats, curr_h)        
            theta = self.fc_theta(curr_h.squeeze(0))
            thetas.append(theta)

            # produce the view
            theta = theta.view(-1, 2, 3)
            output_size = torch.Size([x.shape[0], *self.input_shape])
            grid = F.affine_grid(theta, output_size, align_corners=False)
            view = F.grid_sample(x, grid, align_corners=False)
            view = self.transform(view)[0]
            views.append(view)

        return views, thetas

     
    def trans_theta(self, theta):
        return theta


    def dim(self):
        return 6


class ST_AffineDiff(nn.Module):
    def __init__(self, input_shape):
        super(ST_AffineDiff, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta, inverse=False):
        if inverse:
            theta = -theta
        theta = theta.view(-1, 2, 3)
        theta = expm(theta)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x
    
    def trans_theta(self, theta):
        return expm(theta)
    
    def dim(self):
        return 6


if __name__ == '__main__':
    pass