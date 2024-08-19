import os
import optuna
import torch
from model import common
import utility
import template
import data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from option import args
from ptflops import get_model_complexity_info
import time
import torch.nn.utils as utils
import numpy as np


"""
This file is used to initialize all models used in the experiment and calculate their parameter quantities and other parameters
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ABLATION(nn.Module):
    def __init__(self, re,fe, conv=common.default_conv):
        super(ABLATION, self).__init__()

        n_resblocks = re
        n_feats = fe
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)


        # self.sub_mean = common.MeanShift(args.rgb_range)
        # Subtract the mean of the RGB channels from the image.
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # Add the mean of the RGB channels back to the image.
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)


        m_head = [conv(args.n_colors+1, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats,args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.input_conv(x)
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

class EDSR(nn.Module):
    def __init__(self, op_resblocks, op_feats, op_res_scale, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = op_resblocks
        n_feats = op_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # Subtract the mean of the RGB channels from the image.

        # Add the mean of the RGB channels back to the image.
        if args.RCNN_channel == "on":
            channels = args.n_colors + 1
        else:
            channels = args.n_colors
        # define head module
        m_head = [conv(channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=op_res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


class MDSR(nn.Module):
    def __init__(self, RE,FE, conv=common.default_conv):
        super(MDSR, self).__init__()
        n_resblocks = RE
        n_feats = FE
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale_idx = 0
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ])

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = nn.ModuleList([
            common.Upsampler(conv, s, n_feats, act=False) for s in args.scale
        ])

        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

class VDSR(nn.Module):
    def __init__(self, re,fe, conv=common.default_conv):
        super(VDSR, self).__init__()

        n_resblocks = re
        n_feats = fe
        kernel_size = 3


        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, args.n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):

        res = self.body(x)
        res += x


        return x

model = EDSR(30, 290,0.3).to(device)


pre_train = '/home/6c702main/01.15.2024 Archieve/SuperResCT/EDSR-PyTorch-optuna/experiment/ab_Result1/model/model_latest.pt'
load_from = torch.load(pre_train, {'map_location':device})
# model.load_state_dict(checkpoint)
model.load_state_dict(load_from, strict=False)

# Count the number of model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Count the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Count the number of floating-point operations in the statistical model
flops, params = get_model_complexity_info(model, (3, 48, 48), as_strings=True, print_per_layer_stat=True)

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")




