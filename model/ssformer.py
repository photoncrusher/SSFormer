from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
from timm.models.layers import trunc_normal_
from einops import rearrange, reduce
from model.pvt2 import PyramidVisionTransformerImpr, pvt_v2_b0, pvt_v2_b3
from model.mit import MiT
def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

class SSFormer(nn.Module):
    def __init__(
        self,
        *,
        dims = (64, 128, 320, 512),
        heads = (1, 2, 4, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 768,
        num_classes = 1
    ):
        super().__init__()
        W, H = 352, 352
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.pvt_v2 = pvt_v2_b3(in_chans = channels, num_classes = num_classes)

        self.pvt_v2.init_weights("/hdd/quangdd/ssformer/SSFormer/pretrain/pvt_v2_b3.pth")

        self.localemphasis = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, decoder_dim, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2 ** i, mode='bilinear')
        ) for i, dim in enumerate(dims)])

        self.linear = nn.Linear(decoder_dim * 2, decoder_dim)
        self.final_linear = nn.Linear(decoder_dim, num_classes)
        self.final_upsample = nn.Upsample(scale_factor = 4, mode='bicubic')


        
    def forward(self, x):
        layer_outputs = self.pvt_v2(x)
        le_out = None
        for output, le in zip(reversed(layer_outputs), reversed(self.localemphasis)):
            if le_out is None:
                le_out = le(output)
                
            else:
                le_out = torch.cat((le_out, le(output)), dim=1)
                le_out = self.linear(le_out.permute(0,2,3,1))
                le_out = le_out.permute(0,3,1,2)
        le_out = self.final_linear(le_out.permute(0,2,3,1))
        le_out = le_out.permute(0,3,1,2)
        le_out = self.final_upsample(le_out)
        le_out = torch.sigmoid(le_out)
        return le_out
