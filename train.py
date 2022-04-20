import tqdm
from model import ssformer
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F
from utils.dataloader import DataLoaderSegmentation
import datetime
from torch.utils.data import DataLoader
from cfg import *
from torch.utils.tensorboard import SummaryWriter
model = ssformer.SSFormer().cuda(gpu_device)

class bce_dice_loss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(bce_dice_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.RandomSizedCrop((300, 512), 352, 352),
        A.RandomScale((-0.7, 0.7), p=0.5, interpolation=1),
        A.PadIfNeeded(352, 352, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(352, 352, cv2.INTER_NEAREST),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
loss_fn = bce_dice_loss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay_rate)

TRAIN_PATH = "/hdd/quangdd/src/dataset_pranet"
train_dataset = DataLoaderSegmentation(TRAIN_PATH, transform=train_transform)
training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    with tqdm.tqdm(training_loader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            
            '''
            For embed
            '''
            # k, v = data.items()
            # t, inputs = k
            # m, labels = v
            # inputs = inputs.to(torch.float32)
            # labels = labels.to(torch.float32)

            '''
            from image
            '''
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            # inputs = inputs.permute(0,3,1,2)
            # labels = labels.permute(0,3,1,2)


            inputs = inputs.cuda(gpu_device)
            labels = labels.cuda(gpu_device)
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # print(inputs.shape)
            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % len(tepoch) == len(tepoch) - 1:
                last_loss = running_loss / len(tepoch) # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
            # time.sleep(0.1)
            # print(i)
            # last_loss = running_loss / batch_size

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 200

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    # running_vloss = 0.0
    # for i, vdata in enumerate(validation_loader):
    #     vinputs, vlabels = vdata
    #     voutputs = model(vinputs)
    #     vloss = loss_fn(voutputs, vlabels)
    #     running_vloss += vloss

    # avg_vloss = running_vloss / (i + 1)
    print('LOSS train {}'.format(avg_loss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training Loss',
                    { 'Training' : avg_loss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_loss < best_vloss:
        best_vloss = avg_loss
        model_path = '/home/quangdd/result_ssformer/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1