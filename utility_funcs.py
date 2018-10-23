# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from PIL import Image
import json
import numpy as np

# After first review a replaced and edited some code with inspiration from 
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# Site provided in my reiew

PT_Networks = {"vgg16":25088,
               "densenet121":1024}

def load_data(data_dir):
    '''
    Arguments : The path to the data folder
    Returns : The loaders for the train, validation and test datasets
    
    '''
    data_dir = data_dir 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    
    return trainloader , validloader, testloader, train_data


def network_setup(structure, dropout, hidden_layer1, learning_rate, processing_unit):
    '''
    Arguments: The architecture for the network(vgg16 or densenet121), the hyperparameters for the network (hidden layer 1 nodes, dropout       and learning rate) and whether to use gpu or cpu.
    Returns: The set up model, along with the criterion and the optimizer for the training
    '''
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

        
        classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(PT_Networks[structure], hidden_layer1)),
            ('dropout',nn.Dropout(dropout)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 2048)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(2048, 1024)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(1024, 102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate )
        
        if torch.cuda.is_available() and processing_unit == 'gpu':
            model.cuda()
            print('--Using GPU/Cuda--\n')
        else:
            model.to('cpu')
            print('--Using CPU, Cuda not available-- \n')
        
        print('Model architecture is:', structure)
        print('Structure of the classifier: \n\n', model.classifier)
        print('\n')
        print('Learning rate:', learning_rate, '\n')
        
        return model , optimizer ,criterion 
        
    
def validation(model, validloader, criterion, processing_unit):
    vloss = 0
    accuracy = 0
    for inputs, labels in validloader:

        if torch.cuda.is_available() and processing_unit == 'gpu':
            inputs, labels = inputs.to('cuda') , labels.to('cuda')
        else:
            inputs, labels = inputs.to('cpu') , labels.to('cpu')
        
                    
        outputs = model.forward(inputs)
        vloss = criterion(outputs,labels)
        ps = torch.exp(outputs).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
    return vloss, accuracy

def train_network(model, criterion, optimizer, epochs, print_every, trainloader, validloader, processing_unit):
    '''
    Arguments: The model, criterion, optimizer, number of epochs, number of print_every and processing_unit.
    Returns: Nothing
    
    This function trains the model over a certain number of epochs and displays the training, validation and accuracy every "print_every"       step.
    '''
    steps = 0
    
    print('Number of epochs:' , epochs)
    print('Training of neural network started!')
    for e in range(epochs):
        model.train()
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
        
            if torch.cuda.is_available() and processing_unit == 'gpu':
                inputs,labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu') , labels.to('cpu')
            
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    vloss, accuracy = validation(model, validloader, criterion, processing_unit)
            
               
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(vloss/len(validloader)),
                       "Accuracy: {:.4f}".format(accuracy /len(validloader)))
            
            
                running_loss = 0
                model.train()
            
    print('Training of neural network is now completed!')
    
def check_accuracy_on_test(model, testloader, processing_unit):    
    correct = 0
    total = 0
         
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if torch.cuda.is_available() and processing_unit == 'gpu':
                inputs,labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs,labels = inputs.to('cpu'), labels.to('cpu')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    

def save_checkpoint(save_dir,model,structure, hidden_layer1, dropout, learning_rate, epochs, train_data):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    
    This function saves the model at a user specified path. 
    '''
    
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'learning_rate':learning_rate,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_dir)
    
def load_checkpoint(save_dir):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    if torch.cuda.is_available():
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    learning_rate=checkpoint['learning_rate']
    

    model,_,_ = network_setup(structure, dropout,hidden_layer1, learning_rate, processing_unit='gpu')
    
   
        

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor
    
    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a tensor.
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # Replaced org code with code inspired from https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
    
    img_PIL = Image.open(image_path)
    
    # Resize
    if img_PIL.size[0] > img_PIL.size[1]:
        img_PIL.thumbnail((10000, 256))
    else:
        img_PIL.thumbnail((256, 10000))
        
    # Crop 
    left_margin = (img_PIL.width-224)/2
    bottom_margin = (img_PIL.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img_PIL = img_PIL.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img_PIL = np.array(img_PIL)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img_PIL = (img_PIL - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img_PIL = img_PIL.transpose((2, 0, 1))
   
    '''preproc = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])'''
    
    # img_preproc = preproc(img_PIL)
    
    
    return img_PIL


def predict(image_path, model, topk, processing_unit):
    '''
    Arguments: image_path, model, number of predictions and processing unit.
    Returns: The "topk" most probable choices that the network predicts.
    '''
    
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch).type(torch.FloatTensor)
    img_torch = img_torch.unsqueeze_(0) # first argument in passing image tensor to model is batch size.
    
    if torch.cuda.is_available():
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
    
       
    probability = F.softmax(output.data,dim=1)
    
    top_probs, top_labs = probability.topk(topk)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()} # flip the key of the dict
    
    top_labels = [idx_to_class[i] for i in top_labs]
    
    
    return top_probs, top_labels 
    