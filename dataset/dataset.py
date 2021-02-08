import glob
import os
import sys

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    ])


quick_draw_class_map = {0: 'baseball', 1: 'birthday cake', 2: 'broccoli', 
    3: 'animal migration', 4: 'aircraft carrier', 5: 'bat', 
    6: 'binoculars', 7: 'bulldozer', 8: 'boomerang', 9: 'bee', 
    10: 'anvil', 11: 'bear', 12: 'airplane', 13: 'bench', 
    14: 'bird', 15: 'basket', 16: 'bicycle', 17: 'angel', 
    18: 'bucket', 19: 'bridge', 20: 'belt', 21: 'barn', 
    22: 'bread', 23: 'axe', 24: 'book', 25: 'backpack', 
    26: 'bed', 27: 'banana', 28: 'beard', 29: 'beach', 
    30: 'blackberry', 31: 'blueberry', 32: 'basketball', 
    33: 'bottlecap', 34: 'bathtub', 35: 'bowtie', 
    36: 'broom', 37: 'ant', 38: 'The Great Wall of China', 
    39: 'bandage', 40: 'The Eiffel Tower', 41: 'apple', 
    42: 'ambulance', 43: 'baseball bat', 44: 'bracelet', 
    45: 'asparagus', 46: 'alarm clock', 47: 'The Mona Lisa', 
    48: 'arm', 49: 'brain'}



class MnistDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.image_ids = glob.glob(path + '/**/**/*')
        self.labels = [int(data.split('/')[4]) for data in self.image_ids]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_ids[idx])
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class QuickDrawDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.files = glob.glob(path + '/*')
        self.all_x = []
        self.all_y = []
        self.image_ids, self.label_ids = self.load_data()
        self.transform = transform

        self.image_ids = np.vstack(self.image_ids)
        self.label_ids = np.vstack(self.label_ids)
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.image_ids[idx]
        label_id = int(self.label_ids[idx])

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label_id
    
    def load_data(self):
        count = 0
        for file in self.files:
            images = np.load(file)
            images = images.astype('float32') / 255.
            images = images[0:15000, :] # Subset only 15000 data(There are too many!!)
            images = images.reshape(-1, 28, 28)
            self.all_x.append(images)

            label_ids = [count for _ in range(len(images))]
            label_ids = np.array(label_ids).astype('float32')
            label_ids = label_ids.reshape(label_ids.shape[0], 1)
            self.all_y.append(label_ids)

            labels = os.path.splitext(os.path.basename(file).split('_')[-1])[0]
            count += 1
            
        return self.all_x, self.all_y

