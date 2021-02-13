import os
import sys
import glob
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torchvision
from torch.utils.data.dataset import random_split
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from model import LeNet
from dataset.dataset import QuickDrawDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device == 'cuda':
    torch.cuda.manual_seed_all(1234)

learning_rate = 0.005
epochs = 100
batch_size = 128

#data_path = '../dataset/QuickDraw'
data_path  ='/home/ubuntu/hdd_ext/hdd4000/quickdraw_dataset'
print("Train data start")
train_data = QuickDrawDataset(data_path)
print("Train data load")
train_len = int(len(train_data) * 0.8)
valid_len = len(train_data) - train_len
train_set, valid_set = random_split(train_data, [train_len, valid_len])

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

valid_loader = DataLoader(dataset=valid_set,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=True)

print("model starts")
model = LeNet(num_classes=345).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_batch_train = len(train_loader)
total_batch_val = len(valid_loader)

train_loss_log = []
train_accuracy_log = []
top5_train_accuracy_log = []

valid_loss_log = []
valid_accuracy_log = []
top5_valid_accuracy_log = []

for epoch in range(epochs):
    train_loss = 0
    train_correct = 0
    top5_train_correct = 0
    top5_valid_correct = 0
    validation_loss = 0
    validation_correct = 0
    train_total = 0
    valid_total = 0
    model.train()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        _, top5_preds = torch.topk(outputs, 5)

        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += torch.sum(preds == labels)
        
        train_len = len(labels)
        for i in range(train_len):
            if labels[i] in top5_preds[i]:
                top5_train_correct += 1
    
    with torch.no_grad():
        for val_input, val_label in valid_loader:
            model.eval()
            val_input = val_input.to(device)
            val_label = val_label.to(device)
            val_outputs = model(val_input)
            val_loss = criterion(val_outputs, val_label)
            
            _, val_preds = torch.max(val_outputs, 1)
            _, top5_val_preds = torch.topk(val_outputs, 5)

            validation_loss += val_loss.item()
            valid_total += val_label.size(0)
            validation_correct += torch.sum(val_preds == val_label)

            train_len = len(labels)
            for i in range(train_len):
                if val_label[i] in top5_val_preds[i]:
                    top5_valid_correct += 1

    epoch_loss = train_loss // total_batch_train
    epoch_acc = 100 * train_correct // train_total
    top5_epoch_acc = 100 * top5_train_correct // train_total
    top5_val_epoch_acc = 100 * top5_valid_correct // valid_total

    train_loss_log.append(epoch_loss)
    train_accuracy_log.append(epoch_acc)
    top5_train_accuracy_log.append(top5_epoch_acc)

    val_epoch_loss = validation_loss // total_batch_val
    val_epoch_acc = 100 * validation_correct // valid_total
    valid_loss_log.append(val_epoch_loss)
    valid_accuracy_log.append(val_epoch_acc)
    top5_valid_accuracy_log.append(top5_val_epoch_acc)
    
    print("===================================================")
    print(f'[epoch: {epoch + 1}]')
    print(f'training loss: {epoch_loss:.4f}, training accuracy: {epoch_acc:.2f} %(top1) {top5_epoch_acc:.2f}%(top5)')
    print(f'validation loss: {val_epoch_loss:.4f}, validation accuracy: {val_epoch_acc:.2f}%(top1) {top5_val_epoch_acc:.2f}%(top5)')

    if epoch//10 == 0:
        # Save model
        torch.save(model.state_dict(), f'../weight/quickdraw_{epochs}.pth')

