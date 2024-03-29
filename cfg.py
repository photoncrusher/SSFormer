import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations.augmentations import transforms
gpu_device = 1

batch_size = 4 
learning_rate = 1e-4
decay_rate = 1e-1

train_transform = A.Compose(
    [
        A.Resize(352, 352),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
#train_transform = A.Compose([
        #A.RandomRotate90(),
        #A.Flip(),
        #A.HueSaturationValue(),
        #A.RandomBrightnessContrast(),
        #A.Transpose(),
        #A.OneOf([
            #A.RandomCrop(224, 224, p=0.2),
            #A.CenterCrop(224, 224, p=0.2)
        #], p=0.2),
        #A.Resize(352,352),
        #ToTensorV2()

    #])
infer_transform = A.Compose(
    [
        A.Resize(352, 352),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)