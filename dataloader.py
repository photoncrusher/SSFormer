from cv2 import imread
import torch.utils.data as data
import glob
import os
import cv2
import torch
import numpy as np
class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'image','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path))) 

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path)
            data = cv2.resize(data, (352, 352))
            label = cv2.imread(mask_path, 0)
            label = cv2.resize(label, (352, 352))
            # label = np.expand_dims(label, axis=2)
            label = label[:, :, np.newaxis]
            return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)