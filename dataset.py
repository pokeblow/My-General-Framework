import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_path_list = glob.glob(os.path.join(data_path, 'image/*.bmp'))
        self.transform = transform

    def __getitem__(self, index):
        # Data path
        image_path = self.image_path_list[index]
        # Image input
        image = np.array(Image.open(image_path))
        # Normalization to 0-1
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range
        image = normalization(image)
        # Transform
        image = self.transform(Image.fromarray(image))

        return image

    def __len__(self):
        return len(self.image_path_list)

if __name__ == "__main__":
    data_path = 'Data/test_dataset_unsupervised'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor
                                    ])
    image_dataset = Data_Loader(data_path, transform)
    print("Length of Dataset:", len(image_dataset))



