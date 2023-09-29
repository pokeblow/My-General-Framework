import torch
from torch import optim
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.autograd import gradcheck

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

from dataset import Data_Loader
from utils.monitor import *
# from model.cycle_gan import Registration_VGG

def train(net, device, data_path, epochs=40, batch_size=1, lr=0.00001, resize=160):
    # Monitor
    my_monitor = monitor(epochs=epochs, train_loss_name_list=['Gan', 'Pan'], val_loss_name_list=['Gan'], device=device)

    # Data Loading
    transform = transforms.Compose([transforms.Resize((resize, resize)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0, 1)
                                    ])
    train_dataset = Data_Loader(data_path, transform)
    valid_dataset = Data_Loader(data_path, transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    # Optimizer & Loss Function
    optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=False,
                                                           threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                           min_lr=0,
                                                           eps=1e-20
                                                           )
    criterion = nn.MSELoss()
    best_loss = float('inf')

    # 训练epoch
    for epoch in range(epochs):
        my_monitor.train_start(lr=0.0005, train_datasize=16)
        # train
        for image in train_loader:
            my_monitor.input_step_info(train_loss_list=[random.random(), random.random()])

        # valid
        my_monitor.val_start(valid_datasize=5)
        for image in valid_loader:
            my_monitor.input_step_info(val_loss_list=[random.random()])

        my_monitor.epoch_summary()

    my_monitor.summary()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = None
    net.to(device=device)
    data_path = "Data/joint_dataset"
    train(net, device, data_path, epochs=10, batch_size=50, lr=0.01, resize=224)




