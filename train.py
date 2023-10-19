import torch
from torch import optim
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.autograd import gradcheck
from torch.utils.data import Dataset, random_split
from model.backbone.test_model import Test_Model

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import datetime

from dataset import Data_Loader
from utils.monitor import *


def train(net, device, data_path, epochs=40, batch_size=1, lr=0.00001, image_size=256, pretrain=False,
          parameter_path='', version='0_0'):
    time = datetime.datetime.now()
    date_time = time.strftime('%Y%m%d')

    # Monitor
    my_monitor = monitor(epochs=epochs, device=device,
                         train_loss_name_list=['loss_1', 'loss_2'],
                         val_loss_name_list=['loss_1']
                         )
    # Data Loading
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0, 1)
                                    ])

    image_dataset = Data_Loader(data_path, transform)

    # Dataset Random Split
    train_size = int(0.8 * len(image_dataset))
    valid_size = len(image_dataset) - train_size
    train_dataset, valid_dataset = random_split(image_dataset, [train_size, valid_size])

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

    # Pretrain Set
    if pretrain:
        pretrain_path = '{}_{}_{}_v{}.pth'.format(parameter_path, type(net).__name__, version)
        state_dict = torch.load(pretrain_path)
        net.module.load_state_dict(state_dict)

    # Start train in epoch
    for epoch in range(epochs):
        my_monitor.train_start(lr=optimizer.state_dict()['param_groups'][0]['lr'], train_datasize=len(train_loader))
        '''
            Start train
        '''
        for image in train_loader:
            net.train()
            image = image.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()

            output = net(image)
            loss = criterion(output, image)

            loss.backward()
            optimizer.step()

            my_monitor.input_step_info(train_loss_list=[loss.item(), loss.item()])

        '''
            Start valid
        '''
        my_monitor.val_start(valid_datasize=len(valid_loader))
        for image in valid_loader:
            net.eval()
            image = image.to(device=device, dtype=torch.float32)
            output = net(image)
            loss = criterion(output, image)

            my_monitor.input_step_info(val_loss_list=[loss.item()])

        my_monitor.epoch_summary(show_image=False)
        # Save parameters
        if my_monitor.get_epoch_summary()['loss_1'] < best_loss:
            best_loss = my_monitor.get_epoch_summary()['loss_1']
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(),
                           './parameter/parallel/best_module_{}_{}_v{}.pth'.format(date_time, type(net).__name__, version))
            else:
                torch.save(net.state_dict(),
                           './parameter/best_module_{}_{}_v{}.pth'.format(date_time, type(net).__name__, version))

        scheduler.step(my_monitor.get_epoch_summary()['loss_1'])

    my_monitor.summary()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Test_Model(in_channels=1, out_channels=1)
    net.to(device=device)
    data_path = "Data/test_dataset_unsupervised"
    pretrain_path = 'parameter/best_module_20231012'
    train(net, device, data_path, epochs=20, batch_size=50, lr=0.01, image_size=224, pretrain=False,
          parameter_path=pretrain_path, version='0_1')
