import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import get_scores
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import argparse

from tqdm import tqdm
from model_simple_resnet import ResNet


from dataset_Generator import DataSetGenerator
from utils import load_config
from torch.utils.data import DataLoader

# config = load_config('hyperparameters/cifar10/4cifar10_get_score-FULL.yaml')
config = load_config('hyperparameters/cifar10/10cifar10-full_EL2N3.yaml')
# config = load_config('hyperparameters/cifar10/10cifar10-full_random.yaml')




train_config = config.get('cifar10',None).train
test_config = config.get('cifar10',None).test
trainset =  DataSetGenerator(train_config)
testset =  DataSetGenerator(test_config)
sampler = None


trainloader =  DataLoader(dataset = trainset,
                                    batch_size = config.batch_size,
                                    # shuffle = True, commented due to sampler (both are mutually exclusive)
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False, # earlier was true
                                    # sampler = sampler,
                                    shuffle = True

                                    )
    
testloader =  DataLoader(dataset = testset,
                                    batch_size = config.batch_size,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False # earlier was true
                                    )





if(config.get_scores):
    config.path_to_save_score = config.path_to_save_score + str(config.random_seed) + '/'
    os.makedirs(config.path_to_save_score,exist_ok=True)


EL2N_score = pd.DataFrame()
GraNd_score = pd.DataFrame()
acc_list = []




device = config.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
print(config.random_seed)
torch.manual_seed(config.random_seed)
net = ResNet()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    if(config.get_scores):
            print('in get scores')
            get_scores(net,trainset,optimizer,criterion,device,EL2N_score,GraNd_score,epoch,config.path_to_save_score)
            # continue


def test(epoch,acc_list):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    print(acc)
    acc_list.append({'epoch':epoch, 'acc': acc})


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch,acc_list)
    scheduler.step()

if(not config.get_scores):
    df = pd.DataFrame(acc_list)
    df.to_csv(f'{config.results_folder}{config.exp_name}{train_config.common_prune_ratio*100}.csv')