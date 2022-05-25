# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:46:30 2021

@author: JerryDai
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision.transforms as transforms

import re
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image


from coatnet import CoAtNet


# In[] def

class AICup_Dataset(Dataset):
    
    def __init__(self, txtdata, transform = None):
        with open(txtdata, 'r') as f:
            imgs, all_label = [], []
            for line in f.readlines():
                line = line.strip().split(" ")
                imgs.append((line[0], line[1]))
                if line[1] not in all_label:
                    all_label.append(line[1])
            classes = set(all_label)
            print("classe number: {}".format(len(classes)))
            classes = sorted(list(classes))
            class_to_idx = {classes[i]: i for i in range(len(classes))}  # convert label to index(from 0 to num_class-1)
            del all_label

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        label = self.class_to_idx[label]
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class dataAugmentation():
    def __init__(self):
        self.data_transforms = {
            "trainImages": transforms.Compose([ 
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
            "testImages": transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
        }
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x

def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model, iterator, optimizer, criterion, scheduler, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.train()
    
    for (x, y) in tqdm(iterator):
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        # y_pred, _ = model(x)
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in tqdm(iterator):

            x = x.to(device)
            y = y.to(device)

            # y_pred, _ = model(x)
            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model, iterator):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            # y_pred, _ = model(x)
            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


# In[] Set Random Seed

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# In[] Generate train.txt val.txt test.txt

ROOT = 'data'

images_dir = os.path.join(ROOT, 'Images')

target_classes = os.listdir(images_dir)

image_path_dict = {}

for root, dirs, files in os.walk(images_dir):
    for f in tqdm(files):
        fullpath = os.path.join(root, f)        
        temp_image_class = re.split('\\\\', fullpath)[2]
        
        image_info = fullpath + ' ' + temp_image_class
        
        if temp_image_class in image_path_dict:
            tmp_list = list(image_path_dict[temp_image_class])
            tmp_list.append(image_info)
            image_path_dict[temp_image_class] = tmp_list 
        else:
            tmp_list = list()
            tmp_list.append(image_info)
            image_path_dict[temp_image_class] = tmp_list

train_list = []
val_list = []
test_list = []

SEED = 4019
train_ratio = 0.9

for i_class in target_classes:
    
    i_image_path_list = image_path_dict[i_class]
    
    total_number = len(i_image_path_list)
    train_number = int(total_number * train_ratio)
    val_number = int(total_number * 0.05)
    
    random.seed(SEED)
    random_list = random.sample(range(0, len(i_image_path_list)), len(i_image_path_list))
    
    train_list += [i_image_path_list[idx] for idx in random_list[:train_number]]
    val_list += [i_image_path_list[idx] for idx in random_list[train_number : train_number + val_number]]
    test_list += [i_image_path_list[idx] for idx in random_list[train_number + val_number:]]
    
    SEED += 1
    
train_list.sort()
val_list.sort()
test_list.sort()
    
with open(os.path.join(ROOT, 'train.txt'), 'w') as file:
        for row in train_list:
            file.write(row + '\n')

with open(os.path.join(ROOT, 'val.txt'), 'w') as file:
        for row in val_list:
            file.write(row + '\n')

with open(os.path.join(ROOT, 'test.txt'), 'w') as file:
        for row in test_list:
            file.write(row + '\n')

# In[]

img_size = 256

train_transforms = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],),
    ])

val_transforms = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_txt = 'data/train.txt'
val_txt = 'data/val.txt'
test_txt = 'data/test.txt'

train_dataset = AICup_Dataset(txtdata = train_txt,
                              transform = train_transforms)  # use LoadMyDataset load img

val_dataset = AICup_Dataset(txtdata = val_txt,
                            transform = val_transforms)

test_dataset = AICup_Dataset(txtdata = test_txt,
                            transform = val_transforms)

train_dataloaders = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 0)
val_dataloaders = DataLoader(val_dataset, batch_size = 16, shuffle = False, num_workers = 0)
test_dataloaders = DataLoader(test_dataset, batch_size = 16, shuffle = True, num_workers = 0)

# train_dataset_sizes = len(train_dataset)
# val_dataset_sizes = len(val_dataset)
# test_dataset_sizes = len(test_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = train_dataset.classes
print(train_dataset.class_to_idx)
num_classes = len(class_names)

# In[]

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
block_types=['C', 'T', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer

model = CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes, block_types=block_types)

check_point = 'tut5-model_256_0.13171063192025895.pt'
if os.path.exists(check_point):
    model.load_state_dict(torch.load(check_point))

# In[]
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

# In[]

FOUND_LR = 1e-4
optimizer = optim.Adam(model.parameters(), lr = FOUND_LR)

# In[]

EPOCHS = 5
STEPS_PER_EPOCH = len(train_dataloaders)
TOTAL_STEPS = (EPOCHS + 1) * (STEPS_PER_EPOCH + 2)

MAX_LRS = [p['lr'] for p in optimizer.param_groups]

scheduler = lr_scheduler.OneCycleLR(optimizer,
                                    max_lr = MAX_LRS,
                                    total_steps = TOTAL_STEPS)

# In[] Train Model

best_valid_loss = float('inf')

epoch_train_loss = []
epoch_valid_loss = []

epoch_train_acc_1 = []
epoch_valid_acc_1 = []

epoch_train_acc_5 = []
epoch_valid_acc_5 = []

print('Start training')
if __name__ == '__main__':
    
    for epoch in range(EPOCHS):
        
        start_time = time.monotonic()
        
        train_loss, train_acc_1, train_acc_5 = train(model, train_dataloaders, optimizer, criterion, scheduler, device)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, val_dataloaders, criterion, device)
        
        epoch_train_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)
        
        epoch_train_acc_1.append(train_acc_1)
        epoch_valid_acc_1.append(valid_acc_1)
        
        epoch_train_acc_5.append(train_acc_5)
        epoch_valid_acc_5.append(valid_acc_5)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            torch.save(model.state_dict(), f'tut5-model_{img_size}_{best_valid_loss}.pt')
    
        end_time = time.monotonic()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
              f'Train Acc @5: {train_acc_5*100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
              f'Valid Acc @5: {valid_acc_5*100:6.2f}%')

# In[] plot loss

    plt.plot(epoch_train_loss, label = 'training')
    plt.plot(epoch_valid_loss, label = 'validation')
    plt.legend()
    plt.title('Training Curve')
    plt.savefig('Training_Curve.png')
    plt.close()

# In[] plot acc

    plt.plot(epoch_train_acc_1, label = 'training')
    plt.plot(epoch_valid_acc_1, label = 'validation')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('Accuracy.png')
    plt.close()


