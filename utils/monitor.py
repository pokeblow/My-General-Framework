import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
import logging

def monitor_view(log_path):
    log_file_path = log_path

    log_data = []

    def str_to_data(data_list):
        for epoch in range(len(data_list)):
            for item in range(len(data_list[epoch])):
                data_list[epoch][item] = float(data_list[epoch][item])

        return data_list

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            line_data = line.strip().split('-')
            data_dic = {'epoch': line_data[0], 'obj': line_data[1], 'data': line_data[2].split(',')}
            log_data.append(data_dic)

    epoch_count = len([item for item in log_data if item["obj"] == "train"])

    lr_line = [item['data'] for item in log_data if item["obj"] == "lr"]
    train_line = [item['data'] for item in log_data if item["obj"] == "train"]
    valid_line = [item['data'] for item in log_data if item["obj"] == "valid"]

    lr_line = str_to_data(lr_line)
    train_line = str_to_data(train_line)
    valid_line = str_to_data(valid_line)

    def summary(epochs, lr_line, train_line, valid_line):
        epoch = range(0, epochs)
        fig = plt.figure()
        lr_line = np.array(lr_line[:epochs])
        plt.plot(epoch, lr_line, '.:', label='lr')
        train_line = np.transpose(np.array(train_line))
        valid_line = np.transpose(np.array(valid_line))
        for item in range(len(train_line)):
            plt.plot(epoch, train_line[item], '.-', label='train loss' + str(item))
        for item in range(len(valid_line)):
            plt.plot(epoch, valid_line[item], '.--', label='valid loss' + str(item))
        plt.xlabel('Epoch')

        plt.grid(True)
        plt.legend()
        plt.show()

    summary(epoch_count, lr_line, train_line, valid_line)


class monitor():
    def __init__(self, epochs, train_loss_name_list=[], val_loss_name_list=[], device='cpu'):
        self.epochs = epochs

        self.epoch_count = 1

        self.lr_list = []

        self.train_loss_monitor = loss_monitor(epochs=self.epochs, steps=0, loss_name_list=train_loss_name_list)
        self.valid_loss_monitor = loss_monitor(epochs=self.epochs, steps=0, loss_name_list=val_loss_name_list)

        self.model = 'Train'

        # Set log file
        log_file_path_step = 'logs/monitor_step.log'
        with open(log_file_path_step, 'w'):
            pass
        file_handler_step = logging.FileHandler(log_file_path_step)
        formatter_step = logging.Formatter('%(asctime)s - %(message)s')
        file_handler_step.setFormatter(formatter_step)

        self.logger_step = logging.getLogger('Logger_step')
        self.logger_step.addHandler(file_handler_step)
        self.logger_step.setLevel(logging.INFO)

        log_file_path_epoch = 'logs/monitor_epoch.log'
        with open(log_file_path_epoch, 'w'):
            pass
        file_handler_epoch = logging.FileHandler(log_file_path_epoch)
        formatter_epoch = logging.Formatter('%(asctime)s - %(message)s')
        file_handler_step.setFormatter(formatter_epoch)

        self.logger_epoch = logging.getLogger('Logger_epoch')
        self.logger_epoch.addHandler(file_handler_epoch)
        self.logger_epoch.setLevel(logging.INFO)

        print('-' * 50)
        print('Device: {}'.format(device))
        current_time = datetime.datetime.now()
        print('Start Time:', current_time)
        print("-" * 50)

    def train_start(self, lr, train_datasize=0):
        self.lr_list.append(lr)
        self.logger_epoch.info('{}-lr-{}'.format(self.epoch_count, lr))
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
            log_info = self.train_loss_monitor.input_step_info(step_loss_list=train_loss_list, printout=printout)
            self.logger_step.info('train - ' + log_info)
        if self.model == 'Valid':
            log_info = self.valid_loss_monitor.input_step_info(step_loss_list=val_loss_list, printout=printout)
            self.logger_step.info('valid - ' + log_info)

    def get_epoch_summary(self):
        return self.train_loss_monitor.get_recent_epoch_summary()

    def epoch_summary(self, printout=True, show_image=False):
        if printout:
            print("-" * 100)
            print('Epoch {} Summary:'.format(self.epoch_count))
            print('.' * 42 + ' Train Summary ' + '.' * 42)
            log_info = self.train_loss_monitor.epoch_summary()
            self.logger_epoch.info('{}-train-{}'.format(self.epoch_count, log_info))
            print('.' * 42 + ' Valid Summary ' + '.' * 42)
            log_info = self.valid_loss_monitor.epoch_summary()
            self.logger_epoch.info('{}-valid-{}'.format(self.epoch_count, log_info))
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

        self.recent_loss_mean = []

        self.loss_monitor_step_tmp = [[] for _ in range(len(loss_name_list))]
        self.loss_monitor_list = {}
        self.loss_step_tmp = {}
        for loss_name in loss_name_list:
            self.loss_monitor_list.setdefault(loss_name, [])
            self.loss_step_tmp.setdefault(loss_name, [])

    def input_step_info(self, step_loss_list=[], printout=True):
        self.step_count += 1
        for index, loss_key in enumerate(self.loss_monitor_list.keys()):
            self.loss_monitor_step_tmp[index].append(step_loss_list[index])

        if printout:
            loss_info = ', '.join(['{}: {:.12f}'.format(loss_key, step_loss_list[index]) for index, loss_key in
                                   enumerate(self.loss_monitor_list.keys())])

            log_info = 'Epoch: {}, Step: {}/{}, '.format(self.epoch_count, self.step_count,
                                                    self.step) + loss_info
            print(log_info)
            return log_info

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

        self.recent_loss_mean = step_mean_list

        loss_mean_info = ', '.join(['{}: {:.12f}'.format(loss_key, step_mean_list[index]) for index, loss_key in
                                    enumerate(self.loss_monitor_list.keys())])
        log_info = ', '.join(['{:.12f}'.format(step_mean_list[index]) for index, loss_key in
                                    enumerate(self.loss_monitor_list.keys())])
        print('Loss Mean \t{}'.format(loss_mean_info))
        loss_var_info = ', '.join(['{}: {:.12f}'.format(loss_key, step_var_list[index]) for index, loss_key in
                                   enumerate(self.loss_monitor_list.keys())])
        print('Loss SD \t{}'.format(loss_var_info))

        self.step_count = 0
        self.epoch_count += 1

        return log_info

    def get_recent_epoch_summary(self):
        for index, loss_key in enumerate(self.loss_monitor_list.keys()):
            self.loss_step_tmp[loss_key] = self.recent_loss_mean[index]
        return self.loss_step_tmp

