from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
from timm.models.layers import trunc_normal_
from einops import rearrange, reduce
from model.pvt2 import pvt_v2_b0
from model.modules.axial_attention import AA_kernel
import math
def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

class SSFormer(nn.Module):
    def __init__(
        self,
        *,
        # dims = (64, 128, 320, 512),
        dims = (32, 64, 160, 256),
        heads = (1, 2, 4, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 1
    ):
        super().__init__()
        W, H = 352, 352
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'
        
        self.localemphasis = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, 1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2 ** i, mode='bilinear')
        ) for i, dim in enumerate(dims)])

        self.linear_1 = nn.Linear(decoder_dim*2, decoder_dim)
        self.linear_2 = nn.Linear(decoder_dim*2, decoder_dim)
        self.linear_3 = nn.Linear(decoder_dim*2, decoder_dim)
        
        self.final_linear = nn.Linear(decoder_dim, num_classes) 
        self.final_upsample = nn.Upsample(scale_factor = 4, mode='bicubic')
        self.dropout = nn.Dropout(0.1)

        self.apply(self._init_weights)
        self.pvt_v2 = pvt_v2_b0(in_chans = channels, num_classes = num_classes)
        self.pvt_v2.init_weights("/hdd/quangdd/ssformer/SSFormer/pretrain/pvt_v2_b0.pth")
    
        self.aa_kernel_1 = AA_kernel(decoder_dim, decoder_dim)
        self.aa_kernel_2 = AA_kernel(decoder_dim, decoder_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


        
    def forward(self, x):
        layer_outputs = self.pvt_v2(x)

        output_raw_3 = layer_outputs[3]
        output_le_3 = self.localemphasis[3](output_raw_3)

        output_raw_2 = layer_outputs[2]
        output_le_2 = self.localemphasis[2](output_raw_2)

        output_raw_1 = layer_outputs[1]
        output_le_1 = self.localemphasis[1](output_raw_1)

        output_raw_0 = layer_outputs[0]
        output_le_0 = self.localemphasis[0](output_raw_0)

        output_cat_23 = torch.cat([output_le_2, output_le_3], dim=1)
        output_cat_23 = self.linear_1(output_cat_23.permute(0,2,3,1)).permute(0,3,1,2)
        output_cat_23 = torch.mul(self.aa_kernel_1(output_cat_23), output_cat_23)
        
        output_cat_123 = torch.cat([output_le_1, output_cat_23], dim=1)
        output_cat_123 = self.linear_2(output_cat_123.permute(0,2,3,1)).permute(0,3,1,2)
        output_cat_123 = torch.mul(self.aa_kernel_2(output_cat_123), output_cat_123)

        output_cat_0123 = torch.cat([output_le_0, output_cat_123], dim=1)
        output_cat_0123 = self.linear_3(output_cat_0123.permute(0,2,3,1)).permute(0,3,1,2)

        output_cat_0123 = self.dropout(output_cat_0123)

        output = self.final_linear(output_cat_0123.permute(0,2,3,1)).permute(0,3,1,2)
        output = self.final_upsample(output)

        output = torch.sigmoid(output)
        return output 
