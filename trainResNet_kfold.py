from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle
import preResnet as pre
import pickle
import postResnet as post
import plotTools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
from torch.autograd import Variable


###Configurations / Settings
tiled = False
modelname = 'mse_loss_512'
new_folds = False
pid_dict = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/labels/prob_pid_dict.pickle'
resolution = 512
lr = 0.01

batch_size = 12
momentum = 0.9
gamma = 0.3
dropout = 0.1
waveparams = []
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
torch.cuda.empty_cache()
class_names = ['B','C','D','E','F','G','Calm','NoVis']
no_epochs = 40
step_size = 7 #when to decay the learning rate
mean = 0.48
std = 0.29 #This is from the 'calc mean'
kfold_filename = 'kfold_train_test_split_ycseca.mat'
multilabel_bool = False
pretrained = True
regression = True #Alternatives are "false" for classification and "true" for regression
train_earlier_layers = False
if train_earlier_layers:
    old_model_dir = '/home/server/pi/homes/aellenso/Research/DeepBeach/resnet_models/ycseca/notiled_512/'
    old_model_name = os.listdir(old_model_dir)
    old_model_name = old_model_name[0]

run_title = str(resolution) + '.lr' + str(lr) + '.ss' + str(step_size) + '.bs' + str(batch_size) + 'mo' + str(momentum) + 'gamma' + str(gamma)
split = 0
model_folder = '/home/server/pi/homes/aellenso/Research/DeepBeach/resnet_models/' + modelname + '/'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    val_loss = []
    val_acc = []

    train_loss = []
    train_acc = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100

    for epoch in range(num_epochs):
        print('For test {}, split {}, Epoch {}/{}'.format(modelname, split, epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            # Iterate over data.
            for inputs, id, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                if regression:
                    labels_tensor = torch.Tensor(inputs.shape[0],len(class_names))
                    for ii,simplex_point in enumerate(labels):
                        labels_tensor[:,ii] = simplex_point
                if not regression:
                   labels = labels.to(device, dtype = torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if multilabel_bool:
                        out_sigmoid = torch.sigmoid(outputs)
                        t = Variable(torch.Tensor([0.5])).cuda()  # establish threshold
                        preds = (out_sigmoid > t).float() * 1

                        pos_weight = (labels == 0).sum(dim = 0)/(labels == 1).sum(dim = 0)
                        pos_weight[pos_weight > 1000] = 20 #threshold if the number was divided by 0
                        #Set the weights to be num of neg examples / num of pos examples
                        criterion.register_buffer('pos_weight',pos_weight.float())
                        loss = criterion(outputs.float(), labels.float())
                    if not regression:
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds.float() == labels.data.float())
                        epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    if regression:
                        loss = criterion(outputs, labels_tensor.cuda())


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()

                # statistics
                running_loss += loss.sum().item() #* inputs.size(0)


            epoch_loss = running_loss / dataset_sizes[phase]


            if phase == 'val':
                val_loss.append(epoch_loss)
                #val_acc.append(epoch_acc)

            if phase == 'train':
                train_loss.append(epoch_loss)
                #train_acc.append(epoch_acc)

            print('{} Loss: {:f} '.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss: # Changed this from the highest accuracy to the lowest loss
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss, val_acc, train_acc, train_loss


#This loads the labels dictionary (for the ArgusDataset) and the labels dataframe (to create partitions) from the matfiles

if multilabel_bool:
    labels_df = pd.read_pickle(pid_df)
    labels_dict = {}
    for pid, label in zip(labels_df.pid, labels_df.label):
        try:
            label = label.split('.')
        except AttributeError:
            continue
        #Make sure that all the images are in the classes we're considering
        label = np.unique(label)
        check_in_class_list = [ll in class_names for ll in label]
        if False in check_in_class_list:
            continue
        zeros = np.zeros((len(class_names)))
        multilabel_label = [class_names.index(ll) for ll in label]
        multilabel_label = np.array(multilabel_label)
        zeros[multilabel_label] = 1


        entry = {pid:zeros}
        labels_dict.update(entry)

if not regression:#Create a labels dictionary and pid list for the specific classes you're interested in:
    labels_df = pd.read_pickle(pid_df)
    pids, labels, labels_dict = pre.createLabelsDict(labels_df, class_names)
    no_class = pre.countClasses(labels_dict)
    weights = [1/np.cbrt(no) for no in no_class]
    class_weights = torch.FloatTensor(weights).cuda()

if regression:#Create a labels dictionary and pid labels of a simplex for the specific classes you're interested in:
    with open(pid_dict, 'rb') as f:
        prob_pid_dict = pickle.load(f)
    pids, labels, labels_dict = pre.createLabelsDict_simplex(prob_pid_dict, class_names)


#Generate the folds
if new_folds:
    #Make sure each fold has an equal representation of each class, so if it's multilabel, you have to map
    #Each combination of class to a new fold
    if multilabel_bool:
        pids = list(labels_dict.keys())
        labels = list(labels_dict.values())

        from sklearn.preprocessing import LabelEncoder
        # Map the multilabels into unique labels so that you can use the stratified k fold
        def get_new_labels(y):
            y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
            return y_new

        unique_labels = get_new_labels(labels)

    else:
        unique_labels = labels


    skf = StratifiedKFold(n_splits = 5, random_state = 1)
    skf.get_n_splits(pids, unique_labels)

    partition = {}
    trainfiles = []
    testfiles = []
    kfold_train_files = []
    kfold_test_files = []
    for train_index,test_index in skf.split(pids,unique_labels):
        trainfiles =[pids[pp] for pp in train_index]
        testfiles = [pids[pp] for pp in test_index]
        kfold_train_files.append(trainfiles)
        kfold_test_files.append(testfiles)

    kfold_file = open(kfold_filename, 'wb')
    kfold_dict = {'trainfiles':kfold_train_files,'testfiles':kfold_test_files}
    pickle.dump(kfold_dict, kfold_file)
    kfold_file.close()

else:

    kfold_file = open(kfold_filename, 'rb')
    kfold_inds = pickle.load(kfold_file)
    kfold_file.close()
    kfold_train_files = kfold_inds['trainfiles']
    kfold_test_files = kfold_inds['testfiles']

partition = {}
partition['train'] = kfold_train_files[0]
partition['val'] = kfold_test_files[0]

#replace 'oblique' with 'rect' for data file names for the partition.
#Because the partitions are originally made with oblique imagery, the same images should be used but have to be called with their appropriate path

if tiled == True:
    import TiledArgusDS
    train_transform = transforms.Compose([transforms.Resize((int(resolution/2),int(resolution/2))),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomVerticalFlip(),
                                #transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize([mean, mean, mean],[std, std, std]),
                        ])


    test_transform = transforms.Compose([transforms.Resize((int(resolution/2), int(resolution/2))),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize([mean, mean, mean],[std, std, std]),
                                    ])


    train_ds = TiledArgusDS.TiledArgusTrainDS(basedir, partition['train'], labels_dict, resolution, transform = train_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

    val_ds = TiledArgusDS.TiledArgusTrainDS(basedir, partition['val'], labels_dict, resolution, transform = test_transform)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

else:
    import ArgusDS
    train_transform = transforms.Compose([transforms.Resize((resolution, resolution)),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize([mean, mean, mean],[std, std, std]),
                        ])


    test_transform = transforms.Compose([transforms.Resize((resolution,resolution)),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize([mean, mean, mean],[std, std, std]),
                                    ])

    train_ds = ArgusDS.ArgusTrainDS(basedir, partition['train'], labels_dict, transform = train_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

    val_ds = ArgusDS.ArgusTrainDS(basedir, partition['val'], labels_dict, transform = test_transform)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

dataloaders = {'train':train_dl, 'val':val_dl}

dataset_sizes = {'train':len(train_ds),'val':len(val_ds)}

#Set up the model, load a new one or decide to load a pretrained one

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(class_names)
if pretrained == True:
    model_conv = models.resnet50(pretrained = True)
    num_ftrs = model_conv.fc.in_features
    nb_classes = len(class_names)
    model_conv.fc = nn.Sequential(nn.Dropout(dropout),nn.Linear(num_ftrs, nb_classes))


else:
    model_conv = models.vgg11()
    model_conv.classifier[6].out_features = 8
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)

model_conv = model_conv.to(device)
if multilabel_bool:
    criterion = nn.BCEWithLogitsLoss()
if not regression:
    criterion = nn.CrossEntropyLoss(class_weights)
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)
if regression:
    criterion = nn.MSELoss(reduction = 'none')
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)



if train_earlier_layers == True:
    model_conv.load_state_dict(torch.load(old_model_dir + old_model_name))

    #This is to unfreeze earlier layers
    for c in list(model_conv.children()):
        for p in c.parameters():
            p.requires_grad = True


    # Parameters of newly constructed modules have requires_grad=True by default
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.



# Decay LR by a factor of 0.1 every 50 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=no_epochs)
#    post.CAM(model_conv, dataloaders, device, class_names, mean, std, '/home/server/pi/homes/aellenso')

#Save model and other info about this run
split_model_name = run_title + 'foldno{0:1d}.pth'.format(split)
torch.save(model_conv.state_dict(), model_folder + split_model_name)

split += 1


