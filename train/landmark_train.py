import os
import sys
import csv
import glob
import math
import time
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import LandmarkDataset, LandmarkDatasetOOF
from models import EfficientNetLandmark, ArcFaceLossAdaptiveMargin, ResNext101Landmark
from metrics import AverageMeter, gap
from utils import GradualWarmupSchedulerV2


print('Start training model...')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--save_interval', default=10000, type=int)
parser.add_argument('--max_iter', default=300000, type=int)
parser.add_argument('--verbose_eval', default=10, type=int)
parser.add_argument('--max_size', default=300, type=int)
parser.add_argument('--num_classes', default=1049, type=int)
args = parser.parse_args()

writer = SummaryWriter()

device = 'cuda'
scaler = torch.cuda.amp.GradScaler()

transforms_train = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ImageCompression(
        quality_lower=99, 
        quality_upper=100),
    A.ShiftScaleRotate(
        shift_limit=0.2, 
        scale_limit=0.2, 
        rotate_limit=10, 
        border_mode=0, p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.2),
    A.Resize(args.max_size, args.max_size),
    A.OneOf([
        A.RandomResizedCrop(args.max_size, args.max_size),
        A.Cutout(
            max_h_size=int(args.max_size*0.4), 
            max_w_size=int(args.max_size*0.4),
            num_holes=1,
            p=0.3),
    ]),
    A.Normalize(
        mean=(0.4452, 0.4457, 0.4464),
        std=(0.2592, 0.2596, 0.2600)),
    ToTensorV2(),
])


print('Loading dataset...')
train_dataset = LandmarkDataset(transforms_train)
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


print('Loading model...')
# model = EfficientNetLandmark(1, args.num_classes)
model = ResNext101Landmark(args.num_classes)
model.cuda()

# If resume activate this line with appropriate weight history
# model.load_state_dict(torch.load('./weight/' + 'ResNext101_448_300_36_100000.pth'))


def criterion(inputs, targets, margins):
    arc_face = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
    loss = arc_face(inputs, targets, args.num_classes)
    return loss

# net = nn.DataParallel(model, device_ids=[0, 1])
# criterion = nn.CrossEntropyLoss()

def train():
    epoch_size = len(train_dataset) // args.batch_size
    num_epochs = math.ceil(args.max_iter / epoch_size)

    df = pd.read_csv('./data/train.csv')
    tmp = np.sqrt(1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = CosineAnnealingWarmRestarts(optimizer, num_epochs-1)
    logger = open('log.txt', 'w')
    iteration = 0
    losses = AverageMeter()
    scores = AverageMeter()
    start_epoch = time.time()
    model.train()
    
    for epoch in range(num_epochs):
        if (epoch+1)*epoch_size < iteration:
            continue
        if iteration == args.max_iter:
            break
        correct = 0
        start_time = time.time()
        input_size = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = targets.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, margins)
            # confs, preds = torch.max(outputs.detach(), dim=1)
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
        
            input_size += inputs.size(0)
            losses.update(loss.item(), inputs.size(0))
            scores.update(gap(preds, confs, targets))
            correct += (preds == targets).float().sum()

            iteration += 1

            writer.add_scalar('train_likelihood', losses.val, iteration)
            writer.add_scalar('validation_mse', scores.val, iteration)

            log = {'epoch': epoch+1, 'iteration': iteration, 'loss': losses.val, 'acc': correct.item()/len(train_dataset), 'gap': scores.val}
            logger.write(str(log) + '\n')
            if iteration % args.verbose_eval == 0:
                print(f'[{epoch+1}/{iteration}] Loss: {losses.val:.4f} Acc: {correct/input_size:.4f} GAP: {scores.val:.4f} LR: {scheduler.get_last_lr()} Time: {time.time() - start_time}')
        
            if iteration % args.save_interval == 0:
                torch.save(model.state_dict(), f'ResNext101_448_300_{epoch+36}_{iteration+100000}.pth')
            
            scheduler.step(epoch + i / len(train_loader))

        print()

    logger.close()
    writer.close()
    print(time.time() - start_epoch)

if __name__ == '__main__':
    train()