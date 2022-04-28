import glob
import os

import cv2
import numpy as np
import torch

def mDice(inputs, targets):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()                            
    dice_score = (2.*intersection + 1)/(inputs.sum() + targets.sum() + 1)
    return dice_score


def count_mdice(old_mask_path, new_mask_path):
    file_list = glob.glob(os.path.join(new_mask_path, "*.*"))

    mDiceScore = 0
    for file in file_list:
        file = os.path.basename(file)
        new_path = os.path.join(new_mask_path, file)
        old_path = os.path.join(old_mask_path, file)
        new = cv2.imread(new_path, 0)
        old = cv2.imread(old_path, 0)
        new = new / 255
        old = old / 255
        new = torch.from_numpy(new).float()
        old = torch.from_numpy(old).float()
        mDiceScore += mDice(new, old)

    mDiceScore = mDiceScore / len(file_list)
    return mDiceScore
# count_mdice()