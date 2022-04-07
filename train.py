from segformer_pytorch import segformer_pytorch
import torch
import torch.nn.functional as F
from dataloader import DataLoaderSegmentation
import datetime
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

model = segformer_pytorch.SSFormer()

def loss_fn(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
TRAIN_PATH = "/hdd/quangdd/src/dataset_pranet"
train_dataset = DataLoaderSegmentation(TRAIN_PATH)
training_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        
        inputs = inputs.permute(0, 3, 1, 2)
        labels = labels.permute(0, 3, 1, 2)
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
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

# best_vloss = 1_000_000.

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
    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    #     torch.save(model.state_dict(), model_path)

    epoch_number += 1