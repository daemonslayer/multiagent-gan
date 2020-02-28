#!?usr/bin/env python

import os
from functools import reduce
import shutil
from abc import ABC, abstractmethod

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
# TODO: check if new object required per script
summary_writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, config=None, data=None, model=None):
        super(Evaluator, self).__init__()
        self.config = config
        self.data = data
        self.eval_loss = 0
        self.model = model
        self.criterion = None
        ## visualization config
        # self.visualizer = None

    def setConfig(self, config):
        self.config = config
        return True

    def setData(self, data):
        seld.data = data
        return True

    def setModel(self, model):
        self.model = model
        return True

    def setCriterion(self, criterion):
        self.criterion = criterion
        return True

    def load_saved_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            path = os.path.join(self.config.checkpoints['loc'], \
                    self.config.checkpoints['ckpt_fname'])
        else:
            path = os.path.join(self.config.checkpoints['loc'], checkpoint)
        torch.load(path)

        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints['ckpt_fname'], checkpoint['epoch']))
        return (start_epoch, best_prec1)

    @abstractmethod
    def evaluate(self, epoch):
        pass

class GeneratorEvaluator(Evaluator):
    def __init__(self, config=None, data=None, model=None):
        super(GeneratorEvaluator, self).__init__()
        self.config = config
        self.data = data
        self.eval_loss = 0
        self.model = model
        self.criterion = None

    def evaluate(self, epoch):
        if self.model is None:
            raise ValueError('[-] No model has been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the model')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')

        self.eval_loss = 0
        correct = 0
        # eval mode
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.data):
                if self.config.gpu:
                    images = images.to(device)
                    labels = labels.to(device)

                # compute output
                output = self.model(images)
                self.eval_loss = self.criterion(output, images).item()

                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]

            self.eval_loss /= len(self.data.dataset)
            summary_writer.add_scalar('eval_loss', self.eval_loss)

            print('\nEval Set: Average loss: {:.4f}\n'.format(self.eval_loss))

            return self.eval_loss

class DiscriminatorEvaluator(Evaluator):
    def __init__(self, config=None, data=None, model=None):
        super(DiscriminatorEvaluator, self).__init__(config, data, model)
        self.config = config
        self.data = data
        self.eval_loss = 0
        self.disc_model = model[0]
        self.gen_model = model[1]
        self.criterion = None

    def evaluate(self, epoch):
        if self.disc_model is None:
            raise ValueError('[-] Discriminator model has been provided')
        if self.gen_model is None:
            raise ValueError('[-] Generator model has been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the model')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')

        self.eval_loss = 0
        correct = 0
        # eval mode
        self.disc_model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.data):
                if self.config.gpu:
                    images = images.to(device)
                    labels = labels.to(device)

                # compute output
                disc_in_output = self.disc_model(images)
                disc_gen_output = self.disc_model(self.gen_model(images))
                self.eval_loss = self.criterion(disc_in_output, disc_gen_output)

                # get the index of the max log-probability
                pred = disc_in_output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

            self.eval_loss /= len(self.data.dataset)
            summary_writer.add_scalar('eval_loss', self.eval_loss)

            print('\nEval Set: Average loss: {:.4f}\n'.format(self.eval_loss))

            return self.eval_loss