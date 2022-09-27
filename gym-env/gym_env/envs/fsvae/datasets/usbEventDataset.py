import numpy as np
from  torch.utils.data import Dataset
import os
import torch
from skimage import io
from PIL import Image

class USBEventDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_info, root_dir, split=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = data_info
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.train_split_size = int(0.8 * self.get_size())

    def get_size(self):
        _, _, files = next(os.walk(self.root_dir))
        return len(files)

    def __len__(self):
        self.size = self.get_size()

        if self.split == 'train':
            self.size = self.train_split_size
        elif self.split == 'test':
            self.size -= self.train_split_size

        return self.size 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.split == 'train':
            idx = idx
        elif self.split == 'test':
            idx += self.train_split_size

        # for reading images
        image_name = None
        with open(self.data_info) as f:
            content = f.readlines()
            image_name = content[idx].split()[0]

        img_path = os.path.join(self.root_dir, image_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, image

        # for reading events
        # data_name = None
        # with open(self.data_info) as f:
        #     content = f.readlines()
        #     data_name = content[idx].split()[0]

        # data = np.load(os.path.join(self.root_dir, data_name))
        # return data