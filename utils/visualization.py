import torch
import torch.nn as nn
import torch.nn.functional as F
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

def layer_view_single_channel(layer_tensor=None, channel=0):
    layer_images = layer_tensor.cpu().detach().numpy()[0]
    plt.imshow(layer_images[channel])
    plt.axis('off')
    plt.title('Channel {}'.format(channel))
    plt.tight_layout()
    plt.show()

def layer_view(layer_tensor=None, channel=0):
    layer_images = layer_tensor.cpu().detach().numpy()[0]
    channels, _, _ = layer_images.shape

    row = int(np.sqrt(channels))
    col = channels // row

    fig = plt.figure(figsize=(10, 5))
    grid = plt.GridSpec(row, col*2)

    big_ax = fig.add_subplot(grid[:, col:])
    big_ax.imshow(layer_images[channel])
    big_ax.set_title('Channel {}'.format(channel))
    big_ax.axis('off')

    small_axes = [[fig.add_subplot(grid[i, j]) for j in range(col)] for i in range(row)]

    for i in range(row):
        for j in range(col):
            index = i * col + j
            if index < channels:
                small_axes[i][j].imshow(layer_images[index])
                small_axes[i][j].axis('off')

    plt.tight_layout()

    # 显示图形
    plt.show()
