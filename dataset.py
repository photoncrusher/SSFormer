import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvf
import random
from PIL import Image, ImageOps

DATA_PATH = '/hdd/quangdd/src/dataset_pranet/image/'
MASK_PATH = '/hdd/quangdd/src/dataset_pranet/mask/'
TRAIN_NUM = 2000

class HogeDataset(torch.utils.data.Dataset):
    def __init__(self, transform = None, target_transform = None, train = True):
        #transform and target_transform is a non-random transform such as tensorization
        self.transform = transform
        self.target_transform = target_transform


        data_files = glob.glob(DATA_PATH + '/*.png')
        mask_files = glob.glob(MASK_PATH + '/*.png')

        self.dataset = []
        self.maskset = []

        #Import original image
        for data_file in data_files:
            self.dataset.append(Image.open(
                DATA_PATH + os.path.basename(data_file)
            ).resize((352, 352)))

        #Mask image reading
        for mask_file in mask_files:
            self.maskset.append(ImageOps.grayscale(Image.open(
                MASK_PATH + os.path.basename(mask_file)
            ).resize((352, 352))))

        #Divided into training data and test data
        if train:
            self.dataset = self.dataset[:TRAIN_NUM]
            self.maskset = self.maskset[:TRAIN_NUM]
        else:
            self.dataset = self.dataset[TRAIN_NUM+1:]
            self.maskset = self.maskset[TRAIN_NUM+1:]

        # Data Augmentation
        #Random conversion is done here
        self.augmented_dataset = []
        self.augmented_maskset = []
        for num in range(len(self.dataset)):
            data = self.dataset[num]
            mask = self.maskset[num]
            #Random crop
            for crop_num in range(16):
                #Crop position is determined by random numbers
                i, j, h, w = transforms.RandomCrop.get_params(data, output_size=(250,250))
                cropped_data = tvf.crop(data, i, j, h, w)
                cropped_mask = tvf.crop(mask, i, j, h, w)
                
                #rotation(0, 90, 180,270 degrees)
                for rotation_num in range(4):
                    rotated_data = tvf.rotate(cropped_data, angle=90*rotation_num)
                    rotated_mask = tvf.rotate(cropped_mask, angle=90*rotation_num)
                    
                    #Either horizontal inversion or vertical inversion
                    #Invert(horizontal direction)
                    for h_flip_num in range(2):
                        h_flipped_data = transforms.RandomHorizontalFlip(p=h_flip_num)(rotated_data)
                        h_flipped_mask = transforms.RandomHorizontalFlip(p=h_flip_num)(rotated_mask)
                        
                      
                        #Add Data Augmented data
                        self.augmented_dataset.append(h_flipped_data)
                        self.augmented_maskset.append(h_flipped_mask)

        self.datanum = len(self.augmented_dataset)

    #Data size acquisition method
    def __len__(self):
        return self.datanum

    #Data acquisition method
    #Non-random conversion is done here
    def __getitem__(self, idx):
        out_data = self.augmented_dataset[idx]
        out_mask = self.augmented_maskset[idx]

        if self.transform:
            out_data = self.transform(out_data)

        if self.target_transform:
            out_mask = self.target_transform(out_mask)
        # out_data = out_data.resize(352, 352)
        # out_mask = out_mask.resize(88, 88)

        return out_data, out_mask