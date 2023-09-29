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
        self.img_path = glob.glob(os.path.join(data_path, 'moving/*.bmp'))
        self.transform = transform

        # Input ground truth
        with open('Data/test_dataset_2/xy.txt', "r") as f:
            str = f.read()

        str = str.split('x')
        x_y_list = []
        for i in range(1, len(str)):
            xy_str = str[i].split('y')
            xy_m = [-float(xy_str[0]), -float(xy_str[1])]
            x_y_list.append(xy_m)

        self.ground_truth = x_y_list

    def __getitem__(self, index):
        # 数据路径
        moving_path = self.img_path[index]
        fixed_path = moving_path.replace('moving', 'fixed')
        label = self.ground_truth[index]
        # 读取数据
        moving = np.array(Image.open(moving_path))
        fixed = np.array(Image.open(fixed_path))
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range
        moving = normalization(moving)
        fixed = normalization(fixed)

        # plt.imshow(np.abs(moving-fixed))
        # plt.show()

        # transform
        moving = self.transform(Image.fromarray(moving))
        fixed = self.transform(Image.fromarray(fixed))
        # label = torch.tensor(label)

        return moving, fixed

    def __len__(self):
        return len(self.img_path)

if __name__ == "__main__":
    data_path = 'Data/test_dataset'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor
                                    ])
    image_dataset = Data_Loader(data_path, transform)
    print("Length of Dataset:", len(image_dataset))



