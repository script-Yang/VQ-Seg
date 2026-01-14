from copy import deepcopy
import math
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transform import *

class SemiDataset(Dataset):
    def __init__(self, mode, size=None, root_dir=None, nsample=None,image_size=504):
        #self.name = name
        #self.root = root
        self.mode = mode
        self.image_size = (image_size,image_size)
        self.size = size
        self.root_dir = root_dir
        self.data = []  
        # if mode == 'train_l' or mode == 'train_u':
        #     with open(id_path, 'r') as f:
        #         self.ids = f.read().splitlines()
        #     if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
        #         self.ids *= math.ceil(nsample / len(self.ids))
        #         self.ids = self.ids[:nsample]
        # else:
        #     with open('splits/%s/val.txt' % name, 'r') as f:
        #         self.ids = f.read().splitlines()
        img_dir = os.path.join(self.root_dir  , 'image')
        label_dir = os.path.join(self.root_dir , 'mask')    
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.endswith('.png'):
                img_path = os.path.join(img_dir, img_name)
                label_name = img_name.replace('.png', '.npy')
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    self.data.append({"name": img_name, "img_path": img_path, "label_path": label_path})


    def __getitem__(self, idx):
        #id = self.ids[item]
        #img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        data_entry = self.data[idx]
        img_path, label_path, img_name = data_entry["img_path"], data_entry["label_path"], data_entry["name"]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            #mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1])))) 
            label = np.load(label_path).astype(np.uint8)
            # label = label*255
            mask = Image.fromarray(label)
        if self.mode == 'val':
            img = img.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
            img, mask = normalize(img, mask)
            # mask[mask==255]=1
            #return img, mask, id
            return img, mask, img_name

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            # mask[mask==255]=0
            img, mask = normalize(img, mask)
            return img, mask, img_path
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        # ignore_mask[ignore_value==255] = 1
        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        #return len(self.ids)
        return len(self.data)
