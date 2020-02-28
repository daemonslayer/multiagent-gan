#!/usr/bin/env python

import sys
import os.path
import argparse
from argparse import RawTextHelpFormatter
from inspect import getsourcefile

import numpy as np
import yaml
import torch

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from model import Generator, Discriminator
from trainer import Trainer, GeneratorTrainer, DiscriminatorTrainer
from evaluator import Evaluator, GeneratorEvaluator, DiscriminatorEvaluator
from mapper import *

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
parent_dir = parent_dir[:parent_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from utils import *
sys.path.pop(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    with open('config.yaml', 'r') as file:
    	stream = file.read()
    	config_dict = yaml.safe_load(stream)
    	config = mapper(**config_dict)

    gen_model = Generator(config)
    disc_model = Discriminator(config)
    plt.ion()

    if config.distributed:
        gen_model.to(device)
        gen_model = nn.parallel.DistributedDataParallel(gen_model)
        disc_model.to(device)
        disc_model = nn.parallel.DistributedDataParallel(gen_model)
    elif config.gpu:
        gen_model = nn.DataParallel(gen_model).to(device)
        disc_model = nn.DataParallel(disc_model).to(device)
    else: return

    # Data Loading
    train_dataset = torchvision.datasets.MNIST(root=os.path.join(parent_dir, 'data'),
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root=os.path.join(parent_dir, 'data'),
                                              train=False,
                                              transform=transforms.ToTensor())

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
        num_workers=config.data.workers, pin_memory=config.data.pin_memory, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
        num_workers=config.data.workers, pin_memory=config.data.pin_memory)

    if args.train:
        # trainer settings
        gen_trainer = GeneratorTrainer(config.train.generator, train_loader, gen_model)
        gen_criterion = nn.MSELoss().to(device)
        gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=config.evaluate.hyperparameters.lr, 
            weight_decay=config.evaluate.hyperparameters.weight_decay)
        gen_trainer.setCriterion(gen_criterion)
        gen_trainer.setOptimizer(gen_optimizer)

        disc_trainer = DiscriminatorTrainer(config.train.discriminator, train_loader, [disc_model, gen_model])
        disc_criterion = nn.MSELoss().to(device)
        disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=config.evaluate.hyperparameters.lr, 
            weight_decay=config.evaluate.hyperparameters.weight_decay)
        disc_trainer.setCriterion(disc_criterion)
        disc_trainer.setOptimizer(disc_optimizer)
        
        # evaluator settings
        gen_evaluator = GeneratorEvaluator(config.evaluate, val_loader, gen_model)
        gen_evaluator.setCriterion(gen_criterion)
        
        disc_evaluator = DiscriminatorEvaluator(config.evaluate, val_loader, [disc_model, gen_model])
        disc_evaluator.setCriterion(disc_criterion)

    if args.test:
    	pass

    # Turn on benchmark if the input sizes don't vary
    # It is used to find best way to run models on your machine
    cudnn.benchmark = True
    start_epoch = 0
    g_best_precision = 0
    d_best_precision = 0
    g_start_epoch = 0
    d_start_epoch = 0

    # optionally resume from a checkpoint
    if config.train.generator.resume:
        [g_start_epoch, g_best_precision] = gen_trainer.load_saved_checkpoint(checkpoint=None)
    if config.train.discriminator.resume:    
        [d_start_epoch, d_best_precision] = disc_trainer.load_saved_checkpoint(checkpoint=None)

    # change value to test.hyperparameters on testing
    start_epoch = min(g_start_epoch, d_start_epoch)
    for epoch in range(start_epoch, config.train.total_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        if args.train:
            gen_trainer.adjust_learning_rate(epoch)
            gen_trainer.train(epoch + g_start_epoch)
            g_prec = gen_evaluator.evaluate(epoch)
            
            disc_trainer.adjust_learning_rate(epoch)
            disc_trainer.train(epoch + d_start_epoch)
            d_prec = disc_evaluator.evaluate(epoch)

        if args.test:
        	pass

        # remember best prec@1 and save checkpoint
        if args.train:
            g_is_best = g_prec > g_best_precision
            g_best_precision = max(g_prec, g_best_precision)
            gen_trainer.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': gen_model.state_dict(),
                'best_precision': g_best_precision,
                'optimizer': gen_optimizer.state_dict(),
            }, g_is_best, checkpoint=None)

            d_is_best = d_prec > d_best_precision
            d_best_precision = max(d_prec, d_best_precision)
            disc_trainer.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': disc_model.state_dict(),
                'best_precision': d_best_precision,
                'optimizer': disc_optimizer.state_dict(),
            }, d_is_best, checkpoint=None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
	parser.add_argument('--train', type=str2bool, default='1', \
				help='Turns ON training; default=ON')
	parser.add_argument('--test', type=str2bool, default='0', \
				help='Turns ON testing; default=OFF')
	args = parser.parse_args()
	main(args)
