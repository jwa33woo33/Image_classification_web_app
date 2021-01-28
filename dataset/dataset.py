import glob
import sys

from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    ])

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

