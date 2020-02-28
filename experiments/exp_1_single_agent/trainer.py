#!/usr/bin/env python

import os
from functools import reduce
import shutil
from abc import ABC, abstractmethod

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torchvision
import torch.nn as nn

# TODO: check if new object required per script
summary_writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, config=None, data=None, model=None):
        super(Trainer, self).__init__()
        self.config = config
        self.data = data

        self.train_loss = 0
        self.criterion = None
        self.optimizer = None
        self.curr_lr = 0
        self.start_epoch = 0
        self.best_precision = 0
        self.model = model

    def setConfig(self, config):
        self.config = config
        return True

    def setData(self, data):
        self.data = data
        return True

    def setModel(self, model):
        self.model = model
        self.count_parameters()
        return True

    def setCriterion(self, criterion):
        self.criterion = criterion
        return True

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        return True

    def count_parameters(self):
        if self.model is None:
            raise ValueError('[-] No model has been provided')

        self.trainable_parameters = sum(reduce( lambda a, b: a*b, x.size()) for x in self.model.parameters())

    def get_trainable_parameters(self):
        if self.model is not None and self.trainable_parameters == 0:
            self.count_parameters()

        return self.trainable_parameters

    def save_checkpoint(self, state, is_best, checkpoint=None):
        if not os.path.exists(self.config.checkpoints.loc):
            os.makedirs(self.config.checkpoints.loc)
        if checkpoint is None:
            ckpt_path = os.path.join(self.config.checkpoints.loc, self.config.checkpoints.ckpt_fname)
        else:
            ckpt_path = os.path.join(self.config.checkpoints.loc, checkpoint)
        best_ckpt_path = os.path.join(self.config.checkpoints.loc, \
                            self.config.checkpoints.best_ckpt_fname)
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, best_ckpt_path)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.curr_lr = self.config.hyperparameters.lr * (self.config.hyperparameters.lr_decay ** (epoch // self.config.hyperparameters.lr_decay_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.curr_lr
    
    @abstractmethod
    def load_saved_checkpoint(self, checkpoint=None):
        pass

    @abstractmethod
    def train(self, epoch):
        pass

class GeneratorTrainer(Trainer):
    def __init__(self, config=None, data=None, model=None):
        super(GeneratorTrainer, self).__init__(config=None, data=None, model=None)
        self.config = config
        self.data = data

        self.train_loss = 0
        self.criterion = None
        self.optimizer = None
        self.curr_lr = 0
        self.start_epoch = 0
        self.best_precision = 0
        self.model = model

    def load_saved_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            path = os.path.join(self.config.checkpoints.loc, \
                    self.config.checkpoints.ckpt_fname)
            checkpoint = torch.load(path)
        else:
            path = os.path.join(self.config.checkpoints.loc, checkpoint)

        self.start_epoch = checkpoint['epoch']
        self.best_precision = checkpoint['best_precision']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints.ckpt_fname, self.start_epoch))
        return (self.start_epoch, self.best_precision)

    def train(self, epoch):
        if self.model is None:
            raise ValueError('[-] No model has been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the model')
        if self.optimizer is None:
            raise ValueError('[-] Optimizer hasn\'t been mentioned for the model')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')
        self.train_loss = 0
        self.model.train()
        for batch_idx, (images, labels) in enumerate(self.data):
            if self.config.gpu:
                images = images.to(device)
                labels = labels.to(device)

            output = self.model(images)
            loss = self.criterion(output, images)

            self.optimizer.zero_grad()
            loss.backward()
            self.train_loss = loss.item()
            self.optimizer.step()

            summary_writer.add_scalar('generator_train_loss', loss.item())
            if batch_idx % self.config.logs.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(self.data.dataset),
                    100. * batch_idx / len(self.data),
                    loss.item() / len(self.data), self.curr_lr)
                )
            
            grid = torchvision.utils.make_grid(output)
            summary_writer.add_image('generator_output', grid, 0)

        # self.visualizer.add_values(epoch, loss_train=self.train_loss)
        # self.visualizer.redraw()
        # self.visualizer.block()

class DiscriminatorTrainer(Trainer):
    def __init__(self, config=None, data=None, model=None):
        super(DiscriminatorTrainer, self).__init__(config=None, data=None, model=None)
        self.config = config
        self.data = data

        self.train_loss = 0
        self.criterion = None
        self.optimizer = None
        self.curr_lr = 0
        self.start_epoch = 0
        self.best_precision = 0
        self.disc_model = model[0]
        self.gen_model = model[1]

    def load_saved_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            path = os.path.join(self.config.checkpoints.loc, \
                    self.config.checkpoints.ckpt_fname)
            checkpoint = torch.load(path)
        else:
            path = os.path.join(self.config.checkpoints.loc, checkpoint)

        self.start_epoch = checkpoint['epoch']
        self.best_precision = checkpoint['best_precision']
        self.disc_model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints.ckpt_fname, self.start_epoch))
        return (self.start_epoch, self.best_precision)

    def train(self, epoch):
        if self.disc_model is None:
            raise ValueError('[-] Discriminator model has been provided')
        if self.gen_model is None:
            raise ValueError('[-] Generator model has been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the model')
        if self.optimizer is None:
            raise ValueError('[-] Optimizer hasn\'t been mentioned for the model')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')
        self.train_loss = 0
        self.disc_model.train()
        for batch_idx, (images, labels) in enumerate(self.data):
            if self.config.gpu:
                images = images.to(device)
                labels = labels.to(device)

            disc_in_output = self.disc_model(images)
            disc_gen_output = self.disc_model(self.gen_model(images))
            loss = self.criterion(disc_in_output, disc_gen_output)

            self.optimizer.zero_grad()
            loss.backward()
            self.train_loss = loss.item()
            self.optimizer.step()

            summary_writer.add_scalar('discriminator_train_loss', loss.item())
            if batch_idx % self.config.logs.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(self.data.dataset),
                    100. * batch_idx / len(self.data),
                    loss.item() / len(self.data), self.curr_lr)
                )

        # self.visualizer.add_values(epoch, loss_train=self.train_loss)
        # self.visualizer.redraw()
        # self.visualizer.block()