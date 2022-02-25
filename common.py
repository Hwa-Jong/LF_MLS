
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.ops import deform_conv2d


#---------------------------------------------------------------------------------------------------------------------------------
#module
class Feature_Extractor(nn.Module):
    def __init__(self, in_channels, out_channels, bn, act):
        super(Feature_Extractor, self).__init__()        

        self.FE_img2feat = Conv_Block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bn=bn, act=act, stride=1, dilation=1, bias=False)

        self.Level1_1 = Residual_Block(channels=out_channels, kernel_size=3, act=act, bias=False)
        self.Level1_2 = Residual_Block(channels=out_channels, kernel_size=3, act=act, bias=False)

        self.Level2_1 = Residual_Block(channels=out_channels, kernel_size=3, act=act, bias=True)
        self.Level2_2 = Residual_Block(channels=out_channels, kernel_size=3, act=act, bias=True)

        self.Level3_1 = Residual_Block(channels=out_channels, kernel_size=3, act=act, bias=True)
        self.Level3_2 = Residual_Block(channels=out_channels, kernel_size=3, act=act, bias=True)
        
    def forward(self, img):
        x = self.FE_img2feat(img)

        x1_1 = self.Level1_1(x)
        x1_2 = self.Level1_2(x1_1)

        x2_1 = self.Level2_1(x1_2)
        x2_2 = self.Level2_2(x2_1)
        
        x3_1 = self.Level3_1(x2_2)
        x3_2 = self.Level3_2(x3_1)

        return x1_2, x2_2, x3_2

class Feature_Warper(nn.Module):
    def __init__(self, channels, center, bn, act, deformable_groups=1, RCAB_num = 3):
        super(Feature_Warper, self).__init__()

        self.channel = channels
        self.center = center
        self.bn = bn
        self.act = act
        self.deformable_groups = deformable_groups

        self.deformable1 = DeformableConv2d_V2(channels=self.channel, kernel_size=3, deformable_groups=self.deformable_groups)
        self.deformable2 = DeformableConv2d_V2(channels=self.channel, kernel_size=3, deformable_groups=self.deformable_groups)
        self.deformable3 = DeformableConv2d_V2(channels=self.channel, kernel_size=3, deformable_groups=self.deformable_groups)

        mix_feat = []
        mix_feat.append(nn.Conv2d(in_channels=self.channel*3, out_channels=self.channel, kernel_size=1, bias=False))
        for i in range(RCAB_num):
             mix_feat.append(RCAB(n_feat=self.channel, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act))

        self.mix_feat = nn.Sequential(*mix_feat)


    def forward(self, x1, x2, x3, off_feat1, off_feat2, off_feat3):
        b, r, c, h, w = x1.size()

        tar1 = x1[:,self.center,:,:,:].clone()
        tar2 = x2[:,self.center,:,:,:].clone()
        tar3 = x3[:,self.center,:,:,:].clone()

        tar_off1 = off_feat1[:,self.center,:,:,:].clone()
        tar_off2 = off_feat2[:,self.center,:,:,:].clone()
        tar_off3 = off_feat3[:,self.center,:,:,:].clone()

        feats = []
        for i in range(r):
            if i == self.center:
                t1 = self.deformable1.main_conv(x1[:,i,:,:,:])
                t2 = self.deformable2.main_conv(x2[:,i,:,:,:])
                t3 = self.deformable3.main_conv(x3[:,i,:,:,:])
                 
                tar_feat = torch.cat([t1, t2, t3], dim=1)
                tar_feat = self.mix_feat(tar_feat)
                feats.append(tar_feat)                
                continue

            ref1 = x1[:,i,:,:,:].clone()
            ref2 = x2[:,i,:,:,:].clone()
            ref3 = x3[:,i,:,:,:].clone()

            ref_off1 = off_feat1[:,i,:,:,:].clone()
            ref_off2 = off_feat2[:,i,:,:,:].clone()
            ref_off3 = off_feat3[:,i,:,:,:].clone()

            # calculate offset

            feat1 = torch.cat([tar_off1, ref_off1], dim=1)
            aligned_ref1, off1, m1 = self.deformable1(ref1, feat1)
       
            feat2 = torch.cat([tar_off2, ref_off2], dim=1)
            aligned_ref2, off2, m2 = self.deformable2(ref2, feat2)

            feat3 = torch.cat([tar_off3, ref_off3], dim=1)
            aligned_ref3, off3, m3 = self.deformable3(ref3, feat3)

            aligned_feat = torch.cat([aligned_ref1, aligned_ref2, aligned_ref3], dim=1)
            aligned_feat = self.mix_feat(aligned_feat)
            feats.append(aligned_feat)

        x = torch.stack(feats, dim=1) #(b, r, c, h, w)

        return x

class Feature_Mixing(nn.Module): 
    def __init__(self, channels, bn, act, center):
        super(Feature_Mixing, self).__init__()

        self.center = center
        self.act = act

        self.similarity = Similarity(channels=channels, center=center)

        self.RCAB_input = nn.Conv2d(in_channels=channels*8, out_channels=channels*2, kernel_size=1, bias=False)

        self.RCAB1 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)
        self.RCAB2 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)

        self.RCAB3 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)
        self.RCAB4 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)

        self.RCAB5 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)
        self.RCAB6 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)

        self.RCAB7 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)
        self.RCAB8 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)

        self.RCAB9 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)
        self.RCAB10 = RCAB(n_feat=channels*2, kernel_size=3, reduction=4, bias=True, bn=bn, res_scale=1, act=self.act)

    def forward(self, x):
        target = x[:,self.center, ...].clone()
        # Similarity
        x = self.similarity(x)
        x = x.contiguous()


        # FM_RCAB
        refs = []
        r = x.size(1)
        for i in range(r):
            refs.append(x[:,i,...].clone())

        refs = torch.cat(refs, dim=1)
        refs = self.RCAB_input(refs)

        #reference pool(rp)
        rp1 = self.RCAB1(refs)
        rp1 = self.RCAB2(rp1)

        rp2 = self.RCAB3(rp1)
        rp2 = self.RCAB4(rp2)

        rp3 = self.RCAB5(rp2)
        rp3 = self.RCAB6(rp3)

        rp4 = self.RCAB7(rp3)
        rp4 = self.RCAB8(rp4)

        rp5 = self.RCAB9(rp4)
        rp5 = self.RCAB10(rp5)

        ref_pool = torch.stack([rp1, rp2, rp3, rp4, rp5], dim=1)

        return target, ref_pool


class Upscaling(nn.Module):
    def __init__(self, channels, act, scale):
        super(Upscaling, self).__init__()
        
        self.level1_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level1_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level1_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level1_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level2_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level2_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level2_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level2_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level3_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level3_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level3_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level3_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level4_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level4_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level4_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level4_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level5_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level5_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level5_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level5_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.info_pool_conv = nn.Conv2d(in_channels=8*channels, out_channels=channels, kernel_size=1, bias=False)

        self.level6_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level6_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level6_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level6_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level7_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level7_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level7_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level7_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level8_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level8_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level8_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level8_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.level9_in = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, bias=False)
        self.level9_1 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level9_2 = Residual_Block( channels=channels, kernel_size=3, act=act)
        self.level9_3 = Residual_Block( channels=channels, kernel_size=3, act=act)

        self.SR_conv = Conv_Block(in_channels=channels, out_channels=channels*scale*scale, kernel_size=1, bn=False, act=act, stride=1, dilation=1, bias=True)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor=scale)

        #Feature to Y
        self.toY = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1)

    def forward(self, x, ref_pool):
        ref_level1 = ref_pool[:,0,...].clone()
        ref_level2 = ref_pool[:,1,...].clone()
        ref_level3 = ref_pool[:,2,...].clone()
        ref_level4 = ref_pool[:,3,...].clone()
        ref_level5 = ref_pool[:,4,...].clone()

        level1 = self.level1_in(torch.cat([x, ref_level1], dim=1))
        level1 = self.level1_1(level1)
        level1 = self.level1_2(level1)
        level1 = self.level1_3(level1)

        level2 = self.level2_in(torch.cat([level1, ref_level2], dim=1))
        level2 = self.level2_1(level2)
        level2 = self.level2_2(level2)
        level2 = self.level2_3(level2)

        level3 = self.level3_in(torch.cat([level2, ref_level3], dim=1))
        level3 = self.level3_1(level3)
        level3 = self.level3_2(level3)
        level3 = self.level3_3(level3)

        level4 = self.level4_in(torch.cat([level3, ref_level4], dim=1))
        level4 = self.level4_1(level4)
        level4 = self.level4_2(level4)
        level4 = self.level4_3(level4)

        level5 = self.level5_in(torch.cat([level4, ref_level5], dim=1))
        level5 = self.level5_1(level5)
        level5 = self.level5_2(level5)
        level5 = self.level5_3(level5)

        info = torch.cat([x, level1, level2, level3, level4, level5, ref_level5], 1)
        info_pool = self.info_pool_conv(info)

        level6 = self.level6_in(torch.cat([level5, level4, info_pool],1))
        level6 = self.level6_1(level6)
        level6 = self.level6_2(level6)
        level6 = self.level6_3(level6)

        level7 = self.level7_in(torch.cat([level6, level3, info_pool],1))
        level7 = self.level7_1(level7)
        level7 = self.level7_2(level7)
        level7 = self.level7_3(level7)

        level8 = self.level8_in(torch.cat([level7, level2, info_pool],1))
        level8 = self.level8_1(level8)
        level8 = self.level8_2(level8)
        level8 = self.level8_3(level8)

        level9 = self.level9_in(torch.cat([level8, level1, info_pool],1))
        level9 = self.level9_1(level9)
        level9 = self.level9_2(level9)
        level9 = self.level9_3(level9)

        level9 += x

        feat = self.SR_conv(level9)
        feat = self.PixelShuffle(feat)
        res_img = self.toY(feat)

        return res_img

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn, act, stride=1, dilation=1, bias=True):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, padding=(kernel_size//2))
        self.bn = bn
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        x = self.act(x)

        return x

class Residual_Block(nn.Module):
    def __init__(self, channels, kernel_size, act, bias=True, bn=False, res_scale=1):
        super(Residual_Block, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(channels))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Similarity(nn.Module):
    def __init__(self, channels, center, embed_ratio=1, SM_scale=9, similarity_scale=1):
        super(Similarity, self).__init__()

        self.channels = channels
        self.center = center
        self.embed_ratio = embed_ratio
        self.SM_scale = SM_scale
        self.similarity_scale = similarity_scale

        self.embed_tar = nn.Conv2d(in_channels=channels, out_channels=channels//self.embed_ratio, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.embed_ref = nn.Conv2d(in_channels=channels, out_channels=channels//self.embed_ratio, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def forward(self, x):
        r= x.size(1)

        tar = x[:,self.center,:,:,:].clone()
        tar_embed = self.embed_tar(tar)

        feats = []
        similarity = []
        for i in range(r):
            if i == self.center:
                continue
            feats.append(x[:,i,:,:,:].clone())
            ref = x[:,i,:,:,:].clone()
            ref_embed = self.embed_ref(ref)

            # similarity per spatial
            dot_similarity = torch.sum(tar_embed * ref_embed, dim=1) #(b, h, w)
            
            # similarity per feature
            similarity.append(torch.mean(dot_similarity, dim=[1,2])) #(b,)

        similarity = torch.stack(similarity, dim=1) #( b, r, h, w )
        SM_sim = torch.softmax(similarity*self.similarity_scale, dim=1) #( b, r, h, w  )

        SM_sim = SM_sim.unsqueeze(2).unsqueeze(3).unsqueeze(4) #( b, r, 1, 1, 1)

        x = torch.stack(feats, dim=1)

        x = x * SM_sim * self.SM_scale
        return x


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, reduction=16, bias=True, bn=False, res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DeformableConv2d_V2(nn.Module):
    def __init__(self, channels, kernel_size=3, deform_kernel_size=3, deformable_groups=8):
        super(DeformableConv2d_V2, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.deformable_groups = deformable_groups

        # offset
        self.offset_conv = nn.Conv2d(in_channels=self.channels, out_channels=2*self.deformable_groups*self.kernel_size*self.kernel_size, kernel_size=deform_kernel_size, padding=self.kernel_size//2)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # mask
        self.mask_conv = nn.Conv2d(in_channels=self.channels, out_channels=self.deformable_groups*self.kernel_size*self.kernel_size, kernel_size=deform_kernel_size, padding=self.kernel_size//2)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

        self.main_conv = nn.Conv2d(in_channels=self.channels//deformable_groups, out_channels=self.channels, kernel_size=self.kernel_size, padding=self.kernel_size//2)

    def forward(self, x, feat_offset_mask):
        offset = self.offset_conv(feat_offset_mask)
        mask = torch.sigmoid(self.mask_conv(feat_offset_mask))

        x = deform_conv2d(input=x,
                            offset=offset, 
                            weight=self.main_conv.weight, 
                            bias=self.main_conv.bias, 
                            padding=self.kernel_size//2,
                            mask=mask)
        x *= 2
              
        return x, offset, mask
    