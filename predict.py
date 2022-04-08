import cv2
import numpy as np
from dataloader import DataLoaderSegmentation
from segformer_pytorch.segformer_pytorch import SSFormer
import torch
from torch.utils.data import DataLoader

PRETRAIN = "/hdd/quangdd/ssformer/SSFormer/pretrain/model_20220408_162626_0"

model = SSFormer()
model.load_state_dict(torch.load(PRETRAIN))
model.eval()

TRAIN_PATH = "/hdd/quangdd/src/dataset_pranet"
train_dataset = DataLoaderSegmentation(TRAIN_PATH)
training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

img, lb = next(iter(training_loader))
pred = model(torch.tensor(img.permute(0,3,1,2)))

out = pred[0].reshape(88, 88, 1)
out = out.detach().numpy()



lb = lb[0].detach().numpy()

cv2.imwrite('temp.png', out)
cv2.imwrite('temp1.png', lb)