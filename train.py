import tqdm
# from model import gate_inner_former 
from model import boundary_former
import torch
from utils.dataloader import DataLoaderSegmentation
import datetime
from torch.utils.data import DataLoader
from cfg import *
from loss import bce_dice_loss
from infer_cfg import INFER_IMG_PATH, OUTPUT_DIR
from predict import run_inference
from eval import count_mdice
import os
from torch.utils.tensorboard import SummaryWriter
model = boundary_former.SSFormer().cuda(gpu_device)
loss_fn = bce_dice_loss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay_rate)

TRAIN_PATH = "/hdd/quangdd/src/dataset_pranet"
train_dataset = DataLoaderSegmentation(TRAIN_PATH, transform=train_transform)
training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

infer_dataset = DataLoaderSegmentation(INFER_IMG_PATH, transform=infer_transform)
infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    with tqdm.tqdm(training_loader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            '''
            from image
            '''
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()

            inputs = inputs.cuda(gpu_device)
            labels = labels.cuda(gpu_device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % len(tepoch) == len(tepoch) - 1:
                last_loss = running_loss / len(tepoch) # loss per batch
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
            last_loss = last_loss / batch_size

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0
best_vdice = 0.
best_viou = 0.
EPOCHS = 200

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)
    print('LOSS train {}'.format(avg_loss))
    model.eval()
    run_inference(infer_loader, model)
    score, score2 = count_mdice(os.path.join(OUTPUT_DIR,"old_mask"), os.path.join(OUTPUT_DIR,"new_mask"))
    print("Val DICE score {}, val IOU score {}".format(score, score2))
    writer.add_scalars('Training Loss',
                    { 'Training' : avg_loss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if score > best_vdice:
        best_vdice = score
        model_path = '/home/quangdd/result_ssformer/pvtv2-b0-pretrain/model_{}_{}_{}'.format(epoch_number, score, score2)
        torch.save(model.state_dict(), model_path)
    
    # Track best performance, and save the model's state
    elif score2 > best_viou:
        best_viou = score2
        model_path = '/home/quangdd/result_ssformer/pvtv2-b0-pretrain/model_{}_{}_{}'.format(epoch_number, score, score2)
        torch.save(model.state_dict(), model_path)


    epoch_number += 1