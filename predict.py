import cv2
import numpy as np
from dataloader import DataLoaderSegmentation
from segformer_pytorch.segformer_pytorch import SSFormer
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

PRETRAIN = "/home/quangdd/result_ssformer/model_20220413_183255_18"
train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
model = SSFormer()
model.load_state_dict(torch.load(PRETRAIN))
model.eval()

TRAIN_PATH = "/hdd/quangdd/src/dataset_pranet"
train_dataset = DataLoaderSegmentation(TRAIN_PATH, transform=train_transform)
training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

img, lb = next(iter(training_loader))
pred = model(img)

out = pred[0].reshape(256, 256, 1)
out = out.detach().numpy()



lb = lb[0].detach().numpy()

out = out * 255
lb = lb * 255
cv2.imwrite('temp.png', out)
cv2.imwrite('temp1.png', lb)
