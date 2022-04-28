import cv2
import tqdm
from utils.dataloader import DataLoaderSegmentation
from model.ssformer import SSFormer
import torch
from torch.utils.data import DataLoader
import os
from shutil import rmtree, copy2
from infer_cfg import PRETRAIN, INFER_IMG_PATH, OUTPUT_DIR
from cfg import train_transform, gpu_device

# model = SSFormer().cuda(gpu_device)
# model.load_state_dict(torch.load(PRETRAIN))
# model.eval()


# train_dataset = DataLoaderSegmentation(INFER_IMG_PATH, transform=train_transform)
# infer_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

def run_inference(infer_loader, model):
    rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, 'old_mask'))
    os.makedirs(os.path.join(OUTPUT_DIR, 'new_mask'))
    with tqdm.tqdm(infer_loader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            inputs, labels = data
            inputs = inputs.cuda(gpu_device).float()
            labels = labels.cuda(gpu_device).float()
            pred = (model(inputs)>0.5).float()
            out = pred[0].reshape(352, 352, 1)
            out = out.cpu().detach().numpy()
            lb = labels[0].cpu().detach().numpy()
            out = out * 255
            lb = lb * 255
            path_1 = os.path.join(OUTPUT_DIR, 'old_mask', "mask_{}.png".format(i))
            path_2 = os.path.join(OUTPUT_DIR, 'new_mask', "mask_{}.png".format(i))
            cv2.imwrite(path_1, lb)
            cv2.imwrite(path_2, out)
