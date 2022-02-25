import torch
import torch.nn as nn
import torch.nn.functional as F

import common
 
class LF_MLS(nn.Module): 
    def __init__(self, scale):
        super(LF_MLS, self).__init__()

        self.center = 4
        self.refs_num = 8
        self.scale = scale

        self.FE_channels = 32
        self.offset_channel_rate = 2
        self.FW_channels = 32
        self.FM_channels = 32
        self.SR_channels = 32

        self.bn = False
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        #Feature Extractor
        self.FE = common.Feature_Extractor(in_channels=1, out_channels=self.FE_channels, bn=self.bn, act=self.act)
        self.FE_off = common.Feature_Extractor(in_channels=1, out_channels=self.FE_channels//self.offset_channel_rate, bn=self.bn, act=self.act)

        #Feature Warper
        self.FW = common.Feature_Warper(channels=self.FW_channels, center=self.center, bn=self.bn, act=self.act, deformable_groups=1)

        #Similarity check
        self.FM = common.Feature_Mixing(channels=self.FM_channels, bn=self.bn, act=self.act, center=self.center)

        self.Upscaling = common.Upscaling(channels=self.SR_channels, act=self.act, scale=self.scale)

    def forward(self, x):   
        b, r, c, h, w = x.size()
        lr = x[:,self.center].clone()
        lr_up = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)(lr)

        # Feature Extractor
        x = x.contiguous().view(-1, c, h, w)
        x1, x2, x3 = self.FE(x)
        x1 = x1.contiguous().view(b, r, self.FE_channels, h, w)
        x2 = x2.contiguous().view(b, r, self.FE_channels, h, w)
        x3 = x3.contiguous().view(b, r, self.FE_channels, h, w)

        off1, off2, off3 = self.FE_off(x)
        off1 = off1.contiguous().view(b, r, self.FE_channels//self.offset_channel_rate, h, w)
        off2 = off2.contiguous().view(b, r, self.FE_channels//self.offset_channel_rate, h, w)
        off3 = off3.contiguous().view(b, r, self.FE_channels//self.offset_channel_rate, h, w)

        # Feature Warping
        x = self.FW(x1, x2, x3, off1, off2, off3)

        # Feature Mixing
        target, ref_pool = self.FM(x)

        # Upscaling
        res_img = self.Upscaling(target, ref_pool)
        img = torch.add(res_img, lr_up)

        return img
