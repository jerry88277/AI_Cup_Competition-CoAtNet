# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:41:15 2022

@author: Jerry
"""

import os
import re
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

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
            
# In[] plot bar of counts

label = list(image_path_dict.keys())
label_count = [len(image_path_dict[i_label]) for i_label in label]

fig, ax = plt.subplots(figsize = (16, 12))
plt.bar(label, label_count)

for p in ax.patches:
   ax.annotate(p.get_height(), (p.get_x() + p.get_width()/4, p.get_height() + 0.02), fontsize = 12)

plt.title('Counts of 14 labels', fontsize = 16)
plt.xlabel('label', fontsize = 16)
plt.ylabel('count', fontsize = 16)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('count_plot.png')
plt.close()

total = 0
for i_label_count in label_count:
    total += i_label_count







