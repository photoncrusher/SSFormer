import glob
import os

import cv2
import numpy as np
import torch

SMOOTH = 1e-6

def mDice(inputs, targets):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()                            
    dice_score = (2.*intersection + 1)/(inputs.sum() + targets.sum() + 1)
    return dice_score

def mIOU(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = torch.unsqueeze(outputs, 0)
    labels = torch.unsqueeze(labels, 0)

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    
    return iou# Or thresholded.mean() if you are interested in average across the batch

def count_mdice(old_mask_path, new_mask_path):
    file_list = glob.glob(os.path.join(new_mask_path, "*.*"))

    mDiceScore = 0
    mIOUScore = 0
    for file in file_list:
        file = os.path.basename(file)
        new_path = os.path.join(new_mask_path, file)
        old_path = os.path.join(old_mask_path, file)
        new = cv2.imread(new_path, 0)
        old = cv2.imread(old_path, 0)
        mIOUScore += mIOU(torch.from_numpy(old), torch.from_numpy(new))
        new = new / 255
        old = old / 255
        new = torch.from_numpy(new).float()
        old = torch.from_numpy(old).float()
        mDiceScore += mDice(new, old)
    mDiceScore = mDiceScore / len(file_list)
    mIOUScore = mIOUScore / len(file_list)
    return mDiceScore, mIOUScore
# count_mdice()