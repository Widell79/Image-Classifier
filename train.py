# -*- coding: utf-8 -*-

import argparse
import utility_funcs


parser = argparse.ArgumentParser(description='Training script for my image classifier')

parser.add_argument('data_dir', action="store")
parser.add_argument('--gpu', dest="processing_unit", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

args = parser.parse_args()
structure = args.arch
print_every = 10

trainloader, validloader, testloader, train_data = utility_funcs.load_data(args.data_dir)

model, optimizer, criterion = utility_funcs.network_setup(structure, args.dropout ,args.hidden_units, args.learning_rate, args.processing_unit)


utility_funcs.train_network(model, criterion, optimizer, args.epochs, print_every, trainloader, validloader, args.processing_unit)

utility_funcs.save_checkpoint(args.save_dir, model, structure, args.hidden_units, args.dropout, args.learning_rate, args.epochs, train_data)







