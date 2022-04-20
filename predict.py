import cv2
import numpy as np
import tqdm
from utils.dataloader import DataLoaderSegmentation
from model.ssformer import SSFormer
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import os

PRETRAIN = "/home/quangdd/result_ssformer/model_20220420_112429_16"
train_transform = A.Compose(
    [
        A.Resize(352, 352),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
model = SSFormer()
model.load_state_dict(torch.load(PRETRAIN))
model.eval()

TRAIN_PATH = "/home/quangdd/Downloads/TestDataset/CVC-300/"
train_dataset = DataLoaderSegmentation(TRAIN_PATH, transform=train_transform)
training_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


files = glob.glob(os.path.join(TRAIN_PATH, 'old_mask', '*'))
for f in files:
    os.remove(f)
files = glob.glob(os.path.join(TRAIN_PATH, 'new_mask', '*'))
for f in files:
    os.remove(f)

with tqdm.tqdm(training_loader, unit="batch") as tepoch:
    for i, data in enumerate(tepoch):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.float()
        pred = model(inputs)
        out = pred[0].reshape(352, 352, 1)
        out = out.detach().numpy()
        lb = labels[0].detach().numpy()
        out = out * 255
        lb = lb * 255
        cv2.imwrite('/home/quangdd/result/new_mask/'+ str(i) +'.png', out)
        cv2.imwrite('/home/quangdd/result/old_mask/'+ str(i) +'.png', lb)
