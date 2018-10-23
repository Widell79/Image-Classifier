# Image-Classifier
Project code for Udacity's AI Programming with Python Nanodegree program

Project code to train an image classifier to recognize different species of flowers.
We'll be using [this](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) dataset of 102 flower categories. 

## Prerequisites
The Code is written in Python 3.6.6

Packages that are required are: [Numpy](https://http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/) and [Pytorch](https://pytorch.org/).

## To view Jupyter Notebook
Install [Jupyter](https://jupyter.org/) or [Anaconda](https://www.anaconda.com/)

## Command line application

* Train a new network on a dataset with **train.py**
  * Basic usage : **python train.py data_directory**
  * Prints out training loss, validation loss, and validation accuracy as the network trains

* Options:
  * Set directory to save checkpoints: 
  **python train.py data_dir --save_dir save_directory**
  * Choose arcitecture (vgg16 or densenet121 available): **python train.py data_dir --arch "vgg16"**
  * Set hyperparameters: **python train.py data_dir --learning_rate 0.001 --hidden_units 4096 --epochs 8**
  * Use GPU for training: **python train.py data_dir --gpu gpu**
  
* Predict flower name from an image with **predict.py** along with the probability of that name. That is, you'll pass in a single image **/path/to/image** and return the flower name and class probability.

  * Basic usage: **python predict.py /path/to/image checkpoint**
* Options:
  * Return top KK most likely classes: **python predict.py input checkpoint --top_k 3**
  * Use a mapping of categories to real names: **python predict.py input checkpoint --category_names cat_to_name.json**
  * Use GPU for inference: **python predict.py input checkpoint --gpu**
  
## Label mapping - Json file

You'll also need to load in a mapping from category label to category name. You can find this in the file cat_to_name.json. It's a JSON object which you can read in with the json module.

## CPU / GPU
If you have an NVIDIA GPU then you can install [CUDA](https://developer.nvidia.com/cuda-downloads). With Cuda you will be able to train your model using GPU instead of CPU.


