from math import sqrt
from functools import partial
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from segformer_pytorch.pvt2 import PyramidVisionTransformerImpr, DWConv
# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes
class LocalEmphasis(nn.Module):
    def __init__(self, act_layer=nn.ReLU, drop=0., dim=768, dim_out = 64*64*4):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.upsample = nn.Upsample(dim_out)
        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.upsample(x)
        
        return x
        
class SSFormer(nn.Module):
    def __init__(
        self,
        *,
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

        self.pvt_v2 = PyramidVisionTransformerImpr(
            in_chans = channels,
            embed_dims = dims,
            num_heads = heads,
            num_classes= num_classes,
            mlp_ratios = ff_expansion,
            sr_ratios = reduction_ratio,
        )
        # self.LE = LocalEmphasis
        self.localemphasis = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, 224, 1),
            nn.ReLU(),
            nn.Conv2d(224, decoder_dim, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2 ** i)

            # nn.Upsample(int(decoder_dim * decoder_dim * num_classes / 16))
        ) for i, dim in enumerate(dims)])

        self.linear = nn.Linear(decoder_dim * 2, decoder_dim)
        self.linear1 = nn.Linear(decoder_dim, num_classes)
        self.linear2 = nn.Linear(int(H/4), H)
        self.linear3 = nn.Linear(int(W/4), W)
        # self.final_linear = nn.Linear(int( W * H * decoder_dim / 16), int(W * H * num_classes))
        # self.resample = nn.Upsample(scale_factor = 4)
        
    def forward(self, x):
        layer_outputs = self.pvt_v2(x)
        le_out = None
        for output, le in zip(reversed(layer_outputs), reversed(self.localemphasis)):
            # print(output.shape)
            if le_out is None:
                le_out = le(output)
                
            else:
                le_out = torch.cat((le_out, le(output)), dim=1)
                le_out = self.linear(le_out.permute(0,2,3,1))
                le_out = le_out.permute(0,3,1,2)
        le_out = le_out.permute(0,2,3,1)
        le_out = self.linear1(le_out)
        le_out = le_out.permute(0,2,3,1)
        le_out = self.linear2(le_out)
        le_out = le_out.permute(0,2,3,1)
        le_out = self.linear2(le_out)

        # le_out = self.final_linear(le_out)
        # le_out = torch.reshape(le_out, (1,1,352, 352))
        # le_out = torch.reshape(le_out, le_out)
        # le_out = torch.flatten(le_out)
        # self.linear(le_out)
        # le_out = self.resample(le_out)

        return le_out
