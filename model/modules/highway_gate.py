from cv2 import transform
from torch import nn
import torch
class Highway(nn.Module):
    def __init__(self, size, num_layers):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.gate = nn.Linear(size, size) 

        self.transform_layer = nn.Sequential(nn.Linear(size, size), nn.Linear(size, size)) 

        self.conv_mix = nn.Conv2d(size, size, 3, 1, 1)


    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        gated = torch.sigmoid(self.gate(x))
        channel_gated = x.mul(gated)
        transformed = self.transform_layer(x)
        channel_transform = transformed.mul(1-gated)
        out = channel_gated + channel_transform

        return out