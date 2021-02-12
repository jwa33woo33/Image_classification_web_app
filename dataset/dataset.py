import glob
import os
import sys

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from inference import class_dict_extraction

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

'''
#quick_draw_class_map = {0: 'baseball', 1: 'birthday cake', 2: 'broccoli', 
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
'''
#Make dictionary for quick draw class map with file path
path = '/home/ubuntu/hdd_ext/hdd4000/quickdraw_dataset'
quick_draw_class_map = class_dict_extraction(path, 'npy')


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
            print(file)
            images = images.astype('float32') / 255.
            print(len(images))
            images = images[0:5, :] # Subset only 15000 data(There are too many!!)
            print(len(images))
            images = images.reshape(-1, 28, 28)
            print(len(images))
            self.all_x.append(images)
            print(len(images))

            label_ids = [count for _ in range(len(images))]
            label_ids = np.array(label_ids).astype('float32')
            label_ids = label_ids.reshape(label_ids.shape[0], 1)
            self.all_y.append(label_ids)
            labels = os.path.splitext(os.path.basename(file).split('_')[-1])[0]

            #find label id (key) from the labels (values) in dictionary
            #label_ids = list(quick_draw_class_map.keys())[list(quick_draw_class_map.values()).index(labels)]
            #self.all_y.append(label_ids)
            print(label_ids, count)
            count += 1
        return self.all_x, self.all_y


class LandmarkDataset(Dataset):
    def __init__(self, transforms=None):
        self.image_ids = glob.glob('./data/train/**/**/*')
        with open('./data/train.csv') as f:
            labels = list(csv.reader(f))[1:]
            self.labels = {label[0]: int(label[1]) for label in labels}
            print(self.labels)

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = os.path.splitext(os.path.basename(self.image_ids[idx]))[0]
        label = self.labels[label]

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return image, label
