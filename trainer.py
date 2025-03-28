from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class, calculate_classwise_accuracy_CIFAR, get_scores,load_config


import torchvision
import torch
import torch.nn as nn
from torch import optim
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_Generator import DataSetGenerator
from torchvision.models import resnet18,ResNet18_Weights
import pandas as pd
from model import SimpleDLA



class_count = 10


config = load_config('hyperparameters/cifar10/4cifar10_get_score-FULL.yaml')
# config = load_config('hyperparameters/cifar10/10cifar10-full_EL2N3.yaml')
# config = load_config('hyperparameters/cifar10/10cifar10-full_random.yaml')

# model = ResNeXt_Multi_Class(classCount=class_count) 
print(config.random_seed)
torch.manual_seed(config.random_seed)
# model = resnet18()
# model.fc = nn.Linear(model.fc.in_features, 10) 
model = SimpleDLA()

device = 'cuda:1'

model = model.to(device)

# transformations=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# transformations = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)),  # Resize to (224, 224)
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


train_data=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=True)
testset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=False)

trainloader=DataLoader(dataset=train_data,batch_size=64,num_workers = 64,shuffle = True)
testloader=DataLoader(dataset=testset,batch_size=64,num_workers = 64)

EL2N_score = pd.DataFrame()
GraNd_score = pd.DataFrame()

# train_config = config.get('cifar10',None).train
# test_config = config.get('cifar10',None).test
# train_data =  DataSetGenerator(train_config)
# testset =  DataSetGenerator(test_config)
# sampler = None


# trainloader =  DataLoader(dataset = train_data,
#                                     batch_size = config.batch_size,
#                                     # shuffle = True, commented due to sampler (both are mutually exclusive)
#                                     num_workers = 12,
#                                     pin_memory = True,
#                                     drop_last=False, # earlier was true
#                                     # sampler = sampler,
#                                     shuffle = True

#                                     )
    
# testloader =  DataLoader(dataset = testset,
#                                     batch_size = config.batch_size,
#                                     shuffle = True,
#                                     num_workers = 12,
#                                     pin_memory = True,
#                                     drop_last=False # earlier was true
#                                     )




optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=0)

criterion = nn.CrossEntropyLoss()

if(config.get_scores and False):
    config.path_to_save_score = config.path_to_save_score + str(config.random_seed) + '/'
    os.makedirs(config.path_to_save_score,exist_ok=True)

for epoch in range(30):
    model.train(True)

    trainloss=0
    correct=0
    for x,y in tqdm(trainloader):
        # print(x.shape,y.shape,type(x),type(y))
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        
        yhat=model(x)
        loss=criterion(yhat,y)
        
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
        
        
    if(config.get_scores):
            print('in get scores')
            get_scores(model,train_data,optimizer,criterion,device,EL2N_score,GraNd_score,epoch,config.path_to_save_score)
            # continue
    
    accuracy=[]
    running_corrects=0.0
    for x_test,y_test in tqdm(testloader):
        
        x_test,y_test=x_test.to(device),y_test.to(device)
        yhat=model(x_test)
        _,z=yhat.max(1)
        running_corrects += torch.sum(y_test == z)
    
    accuracy.append(running_corrects/len(testset))
    print('Epoch: {} Loss: {} testAccuracy: {}'.format(epoch,(trainloss/len(train_data)),running_corrects/len(testset)))

print(running_corrects/len(testset))
# accuracy=max(accuracy)
print(accuracy)