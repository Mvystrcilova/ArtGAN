import os
import random

from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.transforms import transforms


class ArtImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files = [file for file in os.listdir(img_dir) if file.startswith('img')]
        # self.imgs = []
        print('Loading files')
        # for file in tqdm(files):
        #     img = Image.open(os.path.join(img_dir, file))
        #     if self.transform:
        #         img = self.transform(img)
        #     self.imgs.append(img)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        item = Image.open(os.path.join(self.img_dir, file))
        if self.transform:
            item = self.transform(item)
        return item


def create_augumented_dataset(img_dir, new_data_dir, num_of_augumentations):
    files = [file for file in os.listdir(img_dir) if not file.startswith('.')]
    transform_list = [transforms.RandomInvert(p=0.8), transforms.RandomSolarize(threshold=192), transforms.RandomAutocontrast(p=0.8),
                  transforms.RandomVerticalFlip(p=0.8), transforms.RandomHorizontalFlip(p=0.8),
                  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)]
    for i, file in enumerate(files):
        img = Image.open(os.path.join(img_dir, file))
        img.save(os.path.join(new_data_dir, f'img_{i}_0.jpg'), format='JPEG', quality=100)
        for augumentation in range(num_of_augumentations):
            num_of_augs = np.random.randint(2, len(transform_list), size=1)[0]
            augs = random.sample(transform_list, num_of_augs)
            augumentations = transforms.Compose(augs)
            augumented_img = augumentations(img)
            rotate = np.random.random()
            if rotate < 0.2:
                augumented_img = augumented_img.rotate(90, expand=True)
            elif rotate < 0.4:
                augumented_img = augumented_img.rotate(180, expand=True)
            elif rotate < 0.6:
                augumented_img = augumented_img.rotate(270, expand=True)
            augumented_img.save(os.path.join(new_data_dir, f'img_{i}_{augumentation+1}.jpg'), format='JPEG', quality=100)


if __name__ == '__main__':
    create_augumented_dataset('data/Abstract_gallery', 'data/augumented_gallery', 6)

