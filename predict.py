# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:26:41 2022

@author: Jerry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary

# import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import re
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import copy
from tqdm import tqdm
from math import sqrt
from PIL import Image
from collections import namedtuple

# from model.conv.MBConv import MBConvBlock
# from model.attention.SelfAttention import ScaledDotProductAttention

from coatnet import CoAtNet

# In[] def

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image  

def plot_images(images, labels, classes, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')

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
            # class_to_idx = {classes[i]: i for i in range(len(classes))}  # convert label to index(from 0 to num_class-1)
            class_to_idx = {i: classes[i] for i in range(len(classes))}  # convert label to index(from 0 to num_class-1)
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
        return img, label, img_name

    def __len__(self):
        return len(self.imgs)

class AICup_Dataset_test(Dataset):
    
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
            # class_to_idx = {i: classes[i] for i in range(len(classes))}  # convert label to index(from 0 to num_class-1)
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
        return img, label, img_name

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

images_dir = os.path.join(ROOT, 'test_data')

target_classes = os.listdir(images_dir)

image_path_dict = {}

for root, dirs, files in os.walk(images_dir):
    for f in tqdm(files):
        fullpath = os.path.join(root, f)        
        temp_image_class = re.split('\\\\', fullpath)[2]
        
        image_info = fullpath + ' ' + temp_image_class
        # image_info = fullpath
        
        if temp_image_class in image_path_dict:
            tmp_list = list(image_path_dict[temp_image_class])
            tmp_list.append(image_info)
            image_path_dict[temp_image_class] = tmp_list 
        else:
            tmp_list = list()
            tmp_list.append(image_info)
            image_path_dict[temp_image_class] = tmp_list

test_data_list = []

for i_class in target_classes:
    
    i_image_path_list = image_path_dict[i_class]
    
    test_data_list += i_image_path_list
    
    # train_list += [i_image_path_list[idx] for idx in random_list[:train_number]]
    # val_list += [i_image_path_list[idx] for idx in random_list[train_number : train_number + val_number]]
    # test_list += [i_image_path_list[idx] for idx in random_list[train_number + val_number:]]


with open(os.path.join(ROOT, 'test_data.txt'), 'w') as file:
        for row in test_data_list:
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
test_data_txt = 'data/test_data.txt'

train_dataset = AICup_Dataset(txtdata = train_txt,
                              transform = train_transforms)  # use LoadMyDataset load img

test_data_dataset = AICup_Dataset_test(txtdata = test_data_txt,
                            transform = val_transforms)

test_dataloaders = DataLoader(test_data_dataset, batch_size = 16, shuffle = False, num_workers = 0)

test_dataset_sizes = len(test_data_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = train_dataset.classes
class_dict = train_dataset.class_to_idx
print(class_dict)

num_classes = len(class_names)

# In[]

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
block_types=['C', 'T', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer

model = CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes, block_types=block_types)

check_point = 'tut5-model_256_0.13461846480667075.pt'
if os.path.exists(check_point):
    model.load_state_dict(torch.load(check_point))

# In[]

criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

FOUND_LR = 5e-4
optimizer = optim.Adam(model.parameters(), lr = FOUND_LR)

# images, labels, probs = get_predictions(model, test_dataloaders)
# pred_labels = torch.argmax(probs, 1)

model.eval()
pred_df = pd.DataFrame(columns = ['image_filename', 'label'])
pred_labels = []

with torch.no_grad():

    for (x, y, path) in tqdm(test_dataloaders):
        
        img_name = [re.split('\\\\', i_path)[-1] for i_path in path]
        
        x = x.to(device)
        y_pred = model(x)
        y_prob = F.softmax(y_pred, dim = -1)
        top_preds = y_prob.argmax(1, keepdim = True).cpu().detach().numpy().reshape(-1,)
        top_pred_labels = [class_dict[i_pred] for i_pred in list(top_preds)]
        
        temp_df = pd.DataFrame(columns = ['image_filename', 'label'])
        temp_df['image_filename'] = img_name
        temp_df['label'] = top_pred_labels
        
        pred_df = pred_df.append(temp_df)

pred_df = pred_df.reset_index(drop = True)
pred_df.to_csv('submission.csv', index = False)

# labels = torch.cat(labels, dim = 0)
















