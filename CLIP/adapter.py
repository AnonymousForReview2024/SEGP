import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torch.nn.init as init


class PONO_group(nn.Module):
    def __init__(self, input_size=None, num_groups=1,return_stats=False, affine=False, eps=1e-5):
        super(PONO_group, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        assert input_size[0] % num_groups == 0, "Number of channels must be divisible by num_groups"
        self.group_channels = input_size[0] // num_groups
        self.num_groups = num_groups

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, num_groups, 1, *input_size[1:])).to('cuda')
            self.gamma = nn.Parameter(torch.ones(1, num_groups, 1, *input_size[1:])).to('cuda')
        else:
            self.beta, self.gamma = None, None


    def forward(self, x):
     
        batch_size, channels, *rest = x.size()
        x = x.view(batch_size, self.num_groups, self.group_channels, *rest)
 
        mean = x.mean(dim=2, keepdim=True)
        if self.num_groups==1024:
            std=torch.zeros_like(x)
            x=torch.zeros_like(x)
        else:
            std = (x.var(dim=2, keepdim=True) + self.eps).sqrt()
            x = (x - mean) / std


        if self.affine:
            beta = self.beta.repeat(1, 1, self.group_channels, *([1] * len(rest)))
            gamma = self.gamma.repeat(1, 1, self.group_channels, *([1] * len(rest)))
            x = x * gamma + beta

        return x, mean, std

class MomentShortcut_group(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MomentShortcut_group, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        #return x
        return x.view(x.size(0), -1, x.size(3),x.size(4))


class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        init.xavier_uniform_(self.fc1[0].weight, gain=1.0)
        init.xavier_uniform_(self.fc2[0].weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        for param in self.parameters():
            param.requires_grad = False
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )


    def forward(self, x):
        with torch.no_grad():
            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            x = torch.cat(
                [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)


            x = x + self.image_encoder.positional_embedding.to(x.dtype)

            x = self.image_encoder.patch_dropout(x)
            x = self.image_encoder.ln_pre(x)

            x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i<self.features[0]:
                with torch.no_grad():
                    x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:#
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)



        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        return 0, seg_patch_tokens, det_patch_tokens

class CLIP_Inplanted_groupPNmixAfterConv_groupMaxNensembleOut(nn.Module):
    def __init__(self, clip_model, features,nGroups,bAffine,nMaxN,bBySum):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        for param in self.parameters():
            param.requires_grad = False
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )

        self.nGroups=nGroups
        self.pono = [PONO_group(affine=bAffine, num_groups=2 ** i, input_size=[1024, 17, 17]) for i in range(0, self.nGroups)]
        self.ms = MomentShortcut_group()

        self.nMaxN=nMaxN
        self.bBySum = bBySum

    def pnmix_batchAll_groupMaxNensemble(self,x):
        batch_size = x.size()[0]
        lam = 0.5

        index = np.random.permutation(batch_size)
        mix_list = []
        mix_sum_list=[]

        for i in range(self.nGroups):
            x_input, mean_input, std_input = self.pono[i](x)
            x2_input, mean2_input, std2_input = self.pono[i](x[index, :])
            x1 = self.ms(x_input, mean2_input, std2_input)
            x2 = self.ms(x2_input, mean_input, std_input)
            mix_y = lam * x1 + (1 - lam) * x2
            mix_list.append(mix_y)
            if self.bBySum==1:
                sum_y = torch.sum(mix_y)
            elif self.bBySum == 0:
                sum_y = torch.var(mix_y)
            elif self.bBySum == 2:
                sum_y = torch.max(mix_y)
            elif self.bBySum == 3:
                sum_y = torch.norm(mix_y, p=2, dim=(1, 2, 3))
            else:
                print("bBySum is error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


            mix_sum_list.append(sum_y)
        values, indices = torch.topk(torch.stack(mix_sum_list), k=self.nMaxN)
        selected_mix_tensors = [mix_list[idx] for idx in indices]
        average_mix = torch.mean(torch.stack(selected_mix_tensors), dim=0)

        return average_mix, index




    def forward(self, x,bPnmix):
        with torch.no_grad():
            x = self.image_encoder.conv1(x)
            if bPnmix:
                x, rand_index = self.pnmix_batchAll_groupMaxNensemble(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            x = torch.cat(
                [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)
            x = x + self.image_encoder.positional_embedding.to(x.dtype)

            x = self.image_encoder.patch_dropout(x)
            x = self.image_encoder.ln_pre(x)

            x = x.permute(1, 0, 2)

        
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i < self.features[0]:
                with torch.no_grad():
                    x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:  
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out  

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)



        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        if bPnmix:
            return 0, seg_patch_tokens, det_patch_tokens,rand_index
        else:
            return 0, seg_patch_tokens, det_patch_tokens







