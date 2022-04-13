import torch
from dataloader import DataLoaderSegmentation
from segformer_pytorch import segformer_pytorch

model = segformer_pytorch.SSFormer(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (64, 16, 4, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 256,              # decoder dimension
    num_classes = 4                 # number of segmentation classes
)

x = torch.randn(1, 3, 352, 352)
pred = model(x)
# TRAIN_PATH = "/hdd/quangdd/src/dataset_pranet"

# train_dataset = DataLoaderSegmentation(TRAIN_PATH)
# training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print(pred.shape)