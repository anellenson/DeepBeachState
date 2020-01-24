from __future__ import print_function, division
import sys

#sys.path.append('/home/server/pi/homes/aellenso/opt/mypython/pycharm-debug.egg')
#import pydevd_pycharm
#pydevd_pycharm.settrace('149.171.148.109', port=22)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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

###Data info

#load the labels dataframe

pid_df_dict = {'duck':'labels/duck_daytimex_labels_df.pickle', 'nbn':'nbn_daytimex_labels.pickle'} ############no longer need this, since we're doing it all with dictionaries
#load the validation files
valfilename = 'labels/nbn_daytimex_valfiles.aug_imgs.pickle'
valfile_duck = 'labels/duck_daytimex_valfiles.aug_imgs.pickle'
removed_shoreline_mean = [0.2714, 0.3129, 0.3416]
removed_shoreline_std = [0.3037, 0.3458, 0.3769]
removed_shoreline_histeq_mean = [0.1680, 0.1719, 0.1871]
removed_shoreline_histeq_std = [0.2567, 0.2591, 0.2708]
nbn_gray_mean = [0.4835,0.4835,0.4835]
nbn_gray_std = [0.2652,0.2652,0.2652]
duck_gray_mean = [0.5199, 0.5199, 0.5199]
duck_gray_std = [0.2319, 0.2319, 0.2319]
rgb_mean = [0.4066, 0.4479, 0.4734]
rgb_std = [0.2850, 0.3098, 0.3322]
class_names = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG'] #TO DO change states list to dashes from matfile
for train_site in ['duck']:
    for imgtype in ['orig_gray']:
        for run in range(5,10):
            valfilename = 'labels/{}_valfiles.pickle'.format(train_site)
            pid_df = pid_df_dict[train_site]
            ds_size = 75
            res_height = 256 #height
            res_width = 256 #width
            batch_size = 16
            lr = 0.008
            gray = True #This is a switch for grayscale or not
            momentum = 0.9
            gamma = 0.1
            equalize_classes = True
            no_epochs = 50
            step_size = 15 #when to decay the learning rate
            mean = duck_gray_mean
            std = duck_gray_std
            waveparams = []
            multilabel_bool = False
            pretrained = True
            train_earlier_layers = False

            ##saveout info
            model_name = 'train_on_{}_stretched__run{}'.format(train_site, run)
            basedirs = ['/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/orig_gray/',
                        '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/']
            conf_folder = 'confusion_table_results/{}/'.format(train_site) + model_name[:-5] + '/'
            if not os.path.exists(conf_folder):
                os.mkdir(conf_folder)

            with open(valfilename, 'rb') as f: #change this from a dictionary
                valfile_dict = pickle.load(f)

            # with open(valfile_duck, 'rb') as f:
            #     valfile_duck_dict = pickle.load(f)

            trainfiles = []
            with open('labels/trainfiles_{}.pickle'.format(train_site)) as f:
                list_of_files = pickle.load(f)
                trainfiles = trainfiles + list_of_files

            valfiles = valfile_dict['valfiles'] #+ valfile_duck_dict['valfiles']

            labels_df = pd.read_pickle(pid_df)
            # duck_df = pd.read_pickle(pid_df_duck)
            # labels_df = pd.concat((labels_df,duck_df))
            # labels_df = labels_df[['pid', 'label']]
            # labels_df = labels_df.reset_index()

            pids, labels, labels_dict = pre.createLabelsDict(labels_df, class_names) # remove this and load a labels dictionary


            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################

            def train_model(model, criterion, optimizer, scheduler, num_epochs):
                since = time.time()
                val_loss = []
                val_acc = []

                train_loss = []
                train_acc = []


                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = 0.0

                for epoch in range(num_epochs):
                    torch.cuda.empty_cache()
                    print('For test {}, Epoch {}/{}'.format(model_name, epoch, num_epochs - 1))
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
                            labels = labels.to(device, dtype = torch.int64)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(inputs)
                                if multilabel_bool == True:
                                    out_sigmoid = torch.sigmoid(outputs)
                                    t = Variable(torch.Tensor([0.5])).cuda()  # establish threshold
                                    preds = (out_sigmoid > t).float() * 1

                                    pos_weight = (labels == 0).sum(dim = 0)/(labels == 1).sum(dim = 0)
                                    pos_weight[pos_weight > 1000] = 20 #threshold if the number was divided by 0
                                    #Set the weights to be num of neg examples / num of pos examples
                                    criterion.register_buffer('pos_weight',pos_weight.float())
                                    loss = criterion(outputs.float(), labels.float())
                                else:

                                    _, preds = torch.max(outputs, 1)
                                    loss = criterion(outputs, labels)

                                # backward + optimize only if in training phase
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds.float() == labels.data.float())

                        epoch_loss = running_loss / dataset_sizes[phase]
                        epoch_acc = running_corrects.double() / dataset_sizes[phase]

                        if phase == 'val':
                            val_loss.append(epoch_loss)
                            val_acc.append(epoch_acc)

                        if phase == 'train':
                            train_loss.append(epoch_loss)
                            train_acc.append(epoch_acc)

                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                            phase, epoch_loss, epoch_acc))

                        # deep copy the model
                        if phase == 'val' and epoch_acc > best_acc:
                            best_acc = epoch_acc
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


            import ArgusDS
            train_transform = transforms.Compose([transforms.Resize((res_height, res_width)),
                                        #transforms.RandomHorizontalFlip(),
                                        #transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(20),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize(mean,std),
                                ])


            test_transform = transforms.Compose([transforms.Resize((res_height,res_width)),
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                    transforms.Normalize(mean,std),
                                            ])

            train_ds = ArgusDS.ArgusTrainDS(basedirs, trainfiles, labels_dict, gray = gray, transform = train_transform)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

            val_ds = ArgusDS.ArgusTrainDS(basedirs, valfiles, labels_dict, gray = gray, transform = test_transform)
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
                model_conv.fc = nn.Linear(num_ftrs, nb_classes)
                optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)

            model_conv = model_conv.to(device)
            criterion = nn.CrossEntropyLoss()


            if train_earlier_layers == True:
                model_conv.load_state_dict(torch.load(old_model_dir + old_model_name))

                #This is to unfreeze earlier layers
                # Parameters of newly constructed modules have requires_grad=True by default
                # Observe that only parameters of final layer are being optimized as
                # opposed to before.
                for c in list(model_conv.children()):
                    for p in c.parameters():
                        p.requires_grad = True


            # Decay LR by a factor of gamma every step_size epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

            model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=no_epochs)
            torch.save(model_conv.state_dict(), 'resnet_models/{}/'.format(train_site) + model_name + '.pth')

            #model_conv.load_state_dict(torch.load('resnet_models/' + model_name + '.pth'))
            conf_dt = post.calcConfusion(model_conv, dataloaders, class_names, device, mean, std, labels_df, waveparams, model_name, plotimgs  = False)
            conf_dt.to_pickle(conf_folder + model_name + '.pickle')
            torch.cuda.empty_cache()
