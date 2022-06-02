import torch
import cv2
import os
import glob
import numpy as np
PATH = "/hdd/quangdd/ssformer/SSFormer/result_new_multiclass"
def save_review_image(tensor, img_type, img_name):
    if not os.path.exists(os.path.join(PATH, img_type)):
        os.makedirs(os.path.join(PATH, img_type))
    new_tensor = torch.sum(tensor, dim=1)
    new_tensor = new_tensor/ torch.max(new_tensor)
    new_tensor = new_tensor.cpu().detach().numpy()[0]
    new_tensor = new_tensor * 255
    resized = cv2.resize(new_tensor, (352, 352))
    cv2.imwrite(os.path.join(PATH, img_type, img_name), resized)


def vstack_image():
    if not os.path.exists(os.path.join(PATH, "f_all")):
        os.makedirs(os.path.join(PATH, "f_all"))
    fx = os.path.join(PATH, "fx")
    f2  = os.path.join(PATH, "f2")
    f3  = os.path.join(PATH, "f3")
    f23  = os.path.join(PATH, "f23")
    f123  = os.path.join(PATH, "f123")
    for path in os.listdir(f2):
        try:
            imgx = cv2.imread(os.path.join(fx, path), 1)
            img2 = cv2.imread(os.path.join(f2, path), 1)
            img3 = cv2.imread(os.path.join(f3, path), 1)
            img23 = cv2.imread(os.path.join(f23, path), 1)
            img123 = cv2.imread(os.path.join(f123, path), 1)
            img = np.hstack((imgx, img2,img3,img23,img123))
            cv2.imwrite(os.path.join(PATH, "f_all", path), img)
        except:
            continue