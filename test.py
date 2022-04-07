import torch
from segformer_pytorch import Segformer

model = Segformer(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 256,              # decoder dimension
    num_classes = 4                 # number of segmentation classes
)

x = torch.randn(1, 3, 256, 256)
pred = model(x)
print(pred.shape)