# -*- coding: utf-8 -*-

import numpy as np

import json
import argparse
from os import path

import utility_funcs

parser = argparse.ArgumentParser(description='Prediction of flower')


parser.add_argument('img_path', action="store")
parser.add_argument('--checkpoint', nargs='*', default='./checkpoint.pth', action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', dest="processing_unit", action="store", default="gpu")

args = parser.parse_args()


model = utility_funcs.load_checkpoint(args.checkpoint)


with open(args.category_names, 'r') as json_file:
    label_dict = json.load(json_file)
    
# correct_folder = args.img_path.split('/')[2]
correct_folder = path.split(args.img_path)[0][-1]



correct_flower = label_dict[str(correct_folder)]

 
probs, classes = utility_funcs.predict(args.img_path, model, args.top_k, args.processing_unit)

flowers = [label_dict[index] for index in classes]


print('Predicting the top {} flowers from image.'.format(args.top_k))
print()
i=0
while i < args.top_k:
    print(i+1,"{} with a probability of {:.4f}".format(flowers[i].capitalize(), probs[i]))
    i += 1
    
print()
print('The correct flower is:', correct_flower.capitalize())
print()
