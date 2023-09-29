import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime


class monitor():
    def __init__(self, epochs, train_loss_name_list=[], val_loss_name_list=[], device='cpu'):
        self.epochs = epochs

        self.epoch_count = 1
        self.step_count = 0

        self.lr_list = []

        self.train_loss_monitor = loss_monitor(epochs=self.epochs, steps=0, loss_name_list=train_loss_name_list)
        self.valid_loss_monitor = loss_monitor(epochs=self.epochs, steps=0, loss_name_list=val_loss_name_list)

        self.model = 'Train'

        print('-' * 50)
        print('Device: {}'.format(device))
        current_time = datetime.datetime.now()
        print('Start Time:', current_time)
        print("-" * 50)

    def train_start(self, lr, train_datasize=0):
        self.lr_list.append(lr)
        self.train_loss_monitor.step = train_datasize
        print()
        print("=" * 100)
        print('Train Model \nEpoch: {} / {}, lr: {}'.format(self.epoch_count, self.epochs, lr))
        print("-" * 100)
        self.model = 'Train'

    def val_start(self, valid_datasize=0):
        self.valid_loss_monitor.step = valid_datasize
        print("-" * 100)
        print('Valid Model \nEpoch: {} / {}'.format(self.epoch_count, self.epochs))
        print("-" * 100)
        self.model = 'Valid'

    def input_step_info(self, train_loss_list=[], val_loss_list=[], printout=True):
        if self.model == 'Train':
            self.train_loss_monitor.input_step_info(step_loss_list=train_loss_list, printout=printout)
        if self.model == 'Valid':
            self.valid_loss_monitor.input_step_info(step_loss_list=val_loss_list, printout=printout)

    def epoch_summary(self, printout=True):
        if printout:
            print("-" * 100)
            print('Epoch {} Summary:'.format(self.epoch_count))
            print('.' * 42 + ' Train Summary ' + '.' * 42)
            self.train_loss_monitor.epoch_summary()
            print('.' * 42 + ' Valid Summary ' + '.' * 42)
            self.valid_loss_monitor.epoch_summary()
            print("=" * 100)
        self.epoch_count += 1

    def summary(self, save=False):
        epoch = range(0, self.epochs)
        fig = plt.figure()
        plt.plot(epoch, self.lr_list, '.:', label='lr')
        for index, loss_key in enumerate(self.train_loss_monitor.loss_monitor_list.keys()):
            plt.plot(epoch, self.train_loss_monitor.loss_monitor_list[loss_key], '.-', label='train ' + loss_key)
        for index, loss_key in enumerate(self.valid_loss_monitor.loss_monitor_list.keys()):
            plt.plot(epoch, self.valid_loss_monitor.loss_monitor_list[loss_key], '.--', label='valid ' + loss_key)
        plt.xlabel('Epoch')

        plt.grid(True)
        plt.legend()
        plt.show()

        if save:
            fig.savefig("train_loss.jpg")
            f = open('train_loss.txt', mode='w')
            for i in self.epoch_loss_list:
                f.write(str(i) + '\n')

    # def val_summary(self):


class loss_monitor():
    def __init__(self, epochs, steps, loss_name_list):
        self.epoch = epochs
        self.step = steps

        self.step_count = 0
        self.epoch_count = 1

        self.loss_monitor_step_tmp = [[] for _ in range(len(loss_name_list))]
        self.loss_monitor_list = {}
        for loss_name in loss_name_list:
            self.loss_monitor_list.setdefault(loss_name, [])

    def input_step_info(self, step_loss_list=[], printout=True):
        self.step_count += 1
        for index, loss_key in enumerate(self.loss_monitor_list.keys()):
            self.loss_monitor_step_tmp[index].append(step_loss_list[index])

        if printout:
            loss_info = ', '.join(['{}: {:.12f}'.format(loss_key, step_loss_list[index]) for index, loss_key in
                                   enumerate(self.loss_monitor_list.keys())])
            print('Epoch: {}, Step: {}/{}, '.format(self.epoch_count, self.step_count,
                                                    self.step) + loss_info)

    def epoch_summary(self):
        step_var_list = []
        step_mean_list = []
        for index, loss_key in enumerate(self.loss_monitor_list.keys()):
            step_mean = np.mean(self.loss_monitor_step_tmp[index])
            step_var = np.var(self.loss_monitor_step_tmp[index])
            self.loss_monitor_list[loss_key].append(step_mean)
            step_var_list.append(step_var)
            step_mean_list.append(step_mean)
            self.loss_monitor_step_tmp[index] = []

        loss_mean_info = ', '.join(['{}: {:.12f}'.format(loss_key, step_mean_list[index]) for index, loss_key in
                                    enumerate(self.loss_monitor_list.keys())])
        print('Loss Mean \t{}'.format(loss_mean_info))
        loss_var_info = ', '.join(['{}: {:.12f}'.format(loss_key, step_var_list[index]) for index, loss_key in
                                   enumerate(self.loss_monitor_list.keys())])
        print('Loss SD \t{}'.format(loss_var_info))

        self.step_count = 0
        self.epoch_count += 1

    def show_out(self, image_list):
        print(image)
