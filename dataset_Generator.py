import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from collections import defaultdict
import random
import os
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

class DataSetGenerator:
    def __init__(self,params):
        self.params = params
        dataset = pd.read_csv(params.path)
        self.image_path = params.image_path
        if(params.use_augmented_data):
            self.modified_dataset = pd.read_csv(params.path_modified)
            self.image_path_modified = params.image_path_modified
        self.num_class =params.num_class

        self.dataset = []
        self.num_classes = params.num_class
        self.class_wise_count = [0]*self.num_classes
        self.labels = [] # used for Weighted random Sampling



        # self.transform = transforms.Compose([
        #     transforms.ToTensor()  # Converts image to tensor (C, H, W) in range [0,1]
        #     ])
        
        # self.transform=transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Resize((224, 224)),  # Resize to (224, 224)
        #                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        if(params.stage == 'train'):
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        self.d = defaultdict(list)
        self.set_id = set()


        for i in range(dataset.shape[0]): # make dictionary of type key : label, value: list of samples
            self.d[dataset.iloc[i]['label']].append(dataset.iloc[i])
        
        
        if(params.stage == 'train' and params.cifar_sample): # if cifar_sample is given then pruned the dataset (cifar_sample is used to make it LT type)
            self.get_samples(params.cifar_sample)


        
        for i in self.d.keys(): # get the unique ids of all the sample it will be used in future
            for j in range(len(self.d[i])):
                self.set_id.add(self.d[i][j]['id'])

        if(params.stage == 'train' and params.get('common_prune')):
            print('--------------------- common prune on some matric ---------------------')

            remove_example = set(get_pruned_example_names_function2(params,params.prune_file,pathDatasetFile = params.path,pathImageDirectory = self.image_path ,prune_ratio = params.cifar_prune_ratio,ratio = params.common_prune_ratio,epoch_no = params.epoch_no))
            for i in self.d.keys():
                tmp = []
                for sample in self.d[i]:
                    if(sample['filename'][:-4] not in remove_example):
                        tmp.append(sample)
                self.d[i] = tmp
                print(len(tmp))


            print(len(remove_example),len(self.d[i]))
            #print('--------------------- we are doing common prune ---------------------')
        elif(params.stage == 'train' and params.get('common_prune_random')):
            print('---------------------common random prune ---------------------')

            remove_example = set(get_random_pruned_example_names_function_Full_dataset(params.prune_file,pathDatasetFile = params.path,pathImageDirectory = self.image_path ,prune_ratio = params.cifar_prune_ratio,ratio = params.common_prune_ratio))
            for i in self.d.keys():
                tmp = []
                for sample in self.d[i]:
                    if(sample['filename'][:-4] not in remove_example):
                        tmp.append(sample)
                self.d[i] = tmp
                print(len(tmp))


        elif(params.stage == 'train' and params.prune_file):
            print(f'using {params.prune_file} to prune dataset')
            remove_example = set(get_pruned_example_names_function(params.prune_file,pathDatasetFile = params.path,pathImageDirectory = self.image_path ,prune_ratio = params.cifar_prune_ratio))

            for i in self.d.keys():
                tmp = []
                for sample in self.d[i]:
                    if(sample['filename'][:-4] not in remove_example):
                        tmp.append(sample)
                self.d[i] = tmp


        elif(params.stage == 'train' and params.cifar_prune_ratio):
            self.get_pruned_samples(params.cifar_prune_ratio)

        if(params.stage == 'train' and params.use_augmented_data):
            self.add_augmented_data()

        for i in sorted(self.d.keys()): # get class wise count and add the samples to self.dataset which will be used in __getitem__ function
            self.class_wise_count[i]=len(self.d[i])
            self.dataset.extend(self.d[i])


        for sample in self.dataset:
            self.labels.append(sample['label'].tolist())


        self.n = len(self.dataset)
        print(f'class wise count for {params.stage} data- {self.class_wise_count}')

    def __getitem__(self, index):
        if(self.params.name == 'CIFAR10-LT'):
            return self.__getitem__cifar(index)
        
        if(self.params.name == 'nih'):
            return self.__getitem__nih(index)

        
    def __getitem__cifar(self, index):
        sample = self.dataset[index]
        if('_' in sample['id']):
            image = Image.open(self.image_path_modified + str(sample['id']) + '.png').convert('RGB')

        else:
            image = Image.open(self.image_path + str(sample['id']) + '.png').convert('RGB')
        image = self.transform(image)
        label = int(sample['label'])
        # label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return (image,label)
    


    def __getitem__nih(self, index):
        sample = self.dataset[index]
        image = Image.open(self.image_path + str(sample['id'])).convert('RGB')
        image = self.transform(image)
        label = sample['label']
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)

        return (image,label)
    


    def get_data(self,index):
        image,label =  self.__getitem__(index)
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)
        path = self.dataset[index]['id']
        return (image,label,path)
    

    
    
    def __len__(self):
        return self.n



    def get_samples(self,cifar_sample):
        for i in self.d.keys():
            # self.d[i] = random.sample(self.d[i],cifar_sample[i])
            self.d[i] = self.d[i][:cifar_sample[i]]



    def get_pruned_samples(self,cifar_prune_ratio):
        for i in self.d.keys():
            # self.d[i] = random.sample(self.d[i],int((1-cifar_prune_ratio[i])*len(self.d[i])))
            self.d[i] = self.d[i][:int((1-cifar_prune_ratio[i])*len(self.d[i]))]

    
    def add_augmented_data(self):
        li = [8,9]

        for i in li:
            samples = self.modified_dataset[self.modified_dataset['label']==i]
            # print(samples.shape)
            for j in range(samples.shape[0]):
                id = samples.iloc[j]['id'].split('_')[0]
                if(id in self.set_id):
                    self.d[samples.iloc[j]['label']].append(samples.iloc[i])











def get_pruned_example_names_function(pathPruneFile,pathDatasetFile,pathImageDirectory,prune_ratio ,epoch_no = 6,ratio = 0.2):
    '''
        function will take GraNd score file or EL2N score file as input, and return the names of smaples which are to pruned in DatasetGenerter.
    '''
    listImageLabels = defaultdict(list)

    dataset_dict = {}
    
    fileDescriptor = open(pathDatasetFile, "r")
        
    #---- get into the loop
    line = True
    line = fileDescriptor.readline()

    
    while line:
                
        line = fileDescriptor.readline()
            
            #--- if not empty
        if line:
          
            lineItems = line.split(",")
            # print(lineItems)
            lineItems[-1] = int(lineItems[-1][:-1])
            # imagePath = os.path.join(pathImageDirectory, lineItems[0])
            imagePath = lineItems[0]
            imageLabel = lineItems[-1]
            # listImageLabels[imageLabel].append(imagePath)
            dataset_dict[imagePath] = imageLabel
    fileDescriptor.close()

    
    fileDescriptor = open(pathPruneFile, "r")
        
        #---- get into the loop
    line = fileDescriptor.readline()

    line = fileDescriptor.readline() # added to ignore the first line (column names)

    cnt = 0
        
    while line:
                
        if line:
          
            lineItems = line.split(",")
            lineItems[-1] = lineItems[-1][:-1]

            img_name = [lineItems[0]]
            
            score = [ float(i) for i in lineItems[1:] ]
            if(len(score)==0):
                print(score)

            listImageLabels[dataset_dict[img_name[0]]].append(img_name+score)
            
        line = fileDescriptor.readline()
    fileDescriptor.close()

    li = []
    for k in listImageLabels.keys():
        listImageLabels[k] = sorted(listImageLabels[k],key = lambda i: i[-1])
        li.append([k,len(listImageLabels[k])])
    
    li = sorted(li,key = lambda i: i[1],reverse = True)
    
    for i in range(len(li)):
        k = li[i][0]
        cnt = li[i][1]
        ratio = prune_ratio[i]
    
        listImageLabels[k] = listImageLabels[k][:int(ratio*cnt)]
    
    pruned_image_names = set()

    for k in listImageLabels.keys():
        for image in listImageLabels[k]:
            pruned_image_names.add(image[0])

    return pruned_image_names











def get_common_pruned_example_names_function_Full_dataset(pathPruneFile,pathDatasetFile,pathImageDirectory,prune_ratio ,epoch_no = 6,ratio = 0.2):
    '''
        function will take GraNd score file or EL2N score file as input, and return the names of smaples which are to pruned in DatasetGenerter.
    '''
    # listImageLabels = defaultdict(list)
    listImageLabels = []

    dataset_dict = {}
    
    fileDescriptor = open(pathDatasetFile, "r")
        
    #---- get into the loop
    line = True
    line = fileDescriptor.readline()

    
    while line:
                
        line = fileDescriptor.readline()
            
            #--- if not empty
        if line:
          
            lineItems = line.split(",")
            # print(lineItems)
            lineItems[-1] = int(lineItems[-1][:-1])
            # imagePath = os.path.join(pathImageDirectory, lineItems[0])
            imagePath = lineItems[0]
            imageLabel = lineItems[-1]
            # listImageLabels[imageLabel].append(imagePath)
            dataset_dict[imagePath] = imageLabel
    fileDescriptor.close()

    
    fileDescriptor = open(pathPruneFile, "r")
        
        #---- get into the loop
    line = fileDescriptor.readline()

    line = fileDescriptor.readline() # added to ignore the first line (column names)

    cnt = 0
        
    while line:
                
        if line:
          
            lineItems = line.split(",")
            lineItems[-1] = lineItems[-1][:-1]

            img_name = [lineItems[0]]
            
            score = [ float(i) for i in lineItems[1:] ]
            if(len(score)==0):
                print(score)

            # listImageLabels[dataset_dict[img_name[0]]].append(img_name+score)
            listImageLabels.append(img_name+score)
            
        line = fileDescriptor.readline()
    fileDescriptor.close()
    '''
    li = []
    for k in listImageLabels.keys():
        listImageLabels[k] = sorted(listImageLabels[k],key = lambda i: i[-1])
        li.append([k,len(listImageLabels[k])])
    
    li = sorted(li,key = lambda i: i[1],reverse = True)
    
    for i in range(len(li)):
        k = li[i][0]
        cnt = li[i][1]
        ratio = prune_ratio[i]
    
        listImageLabels[k] = listImageLabels[k][:int(ratio*cnt)]
    '''
    listImageLabels = sorted(listImageLabels,key = lambda i:i[-1])

    listImageLabels = listImageLabels[:int(ratio*len(listImageLabels))]

    
    pruned_image_names = set()

    # for k in listImageLabels.keys():
    #     for image in listImageLabels[k]:
    #         pruned_image_names.add(image[0])
    for image in listImageLabels:
        pruned_image_names.add(image[0])

    return pruned_image_names




def get_random_pruned_example_names_function_Full_dataset(pathPruneFile,pathDatasetFile,pathImageDirectory,prune_ratio ,epoch_no = 6,ratio = 0.2):
    '''
        function will take GraNd score file or EL2N score file as input, and return the names of smaples which are to pruned in DatasetGenerter.
    '''
    # listImageLabels = defaultdict(list)
    listImageLabels = []

    # dataset_dict = {}
    
    fileDescriptor = open(pathDatasetFile, "r")
        
    #---- get into the loop
    line = True
    line = fileDescriptor.readline()

    
    while line:
                
        line = fileDescriptor.readline()
            
            #--- if not empty
        if line:
          
            lineItems = line.split(",")
            # print(lineItems)
            # lineItems[-1] = int(lineItems[-1][:-1])
            # imagePath = os.path.join(pathImageDirectory, lineItems[0])
            imagePath = lineItems[0]
            # imageLabel = lineItems[-1]
            # listImageLabels[imageLabel].append(imagePath)
            # dataset_dict[imagePath] = imageLabel
            listImageLabels.append(imagePath)
    fileDescriptor.close()
    random.shuffle(listImageLabels)
    listImageLabels = listImageLabels[:int(ratio*len(listImageLabels))]
    print(f'pruning randomly {len(listImageLabels)} for full dataset')
    return listImageLabels



def get_pruned_example_names_function2(params,pathPruneFile,pathDatasetFile,pathImageDirectory,prune_ratio ,epoch_no = 6,ratio = 0.2):
    print('using newly written method ')
    path = [
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/42/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/4242/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/21346/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/30214/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/35678/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/54821/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/69105/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/76453/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/98567/EL2N_score_{epoch_no}.csv',
        f'/data/home1/arunsg/gitproject/data-pruning-4/Scores/CIFAR10-FULL2/424242/EL2N_score_{epoch_no}.csv',
    ]

    dataFrame = pd.DataFrame()
    for file in path:
        df = pd.read_csv(file,index_col=0)
        df['sum'] = df.iloc[:,:].sum(axis=1)
        # df['sum'] = df['sum']/(len(df.columns)-2)
        if(dataFrame.empty):
            dataFrame = df[['sum']]
        else:
            dataFrame = pd.DataFrame({'sum': df['sum'] + dataFrame['sum']})

    
    if(params.random_prune):
        dataFrame = dataFrame.sample(frac=1, random_state=42)

    else:
        dataFrame = dataFrame.sort_values(by='sum', ascending=True)
    print(dataFrame)
    numRows = dataFrame.shape[0]
    dataFrame = dataFrame.iloc[:int(ratio*numRows)]
    indexs = dataFrame.index.tolist()

    return indexs


    


