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

resolution = 10
res2 = 10
batch_size = 2
lr = 0.01
add_segment_states = []
#ds_size =
nbn = True
tiled = False
new_folds = True
pid_df = 'labels/matt_equal_labels_165_df.pickle'
momentum = 0.9
gamma = 0.1
dropout = 0.1
ds_size = 50
equalize_classes = True
kfold = False
ds_size = 50
partition_filename = 'nbn_matt_labels_partition.pickle'
run = 0
model_name = 'nbn_images_orig_ratio_dataset_size{}_run{}'.format(ds_size,run)
waveparams = []
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/'
conf_folder = 'confusion_table_results/' + model_name[:-5] + '/'
if not os.path.exists(conf_folder):
    os.mkdir(conf_folder)
class_names = ['Ref','LTT','TBR','RBB','LBT'] #TO DO change states list to dashes from matfile
no_epochs = 30
step_size = 7 #when to decay the learning rate
mean = [0.4079, 0.4493, 0.4717]
std = [0.28, 0.30, 0.32] #This is from the 'calc mean'
kfold_filename = 'kfold_pure_images.mat'
multilabel_bool = False
pretrained = True
train_earlier_layers = False
if train_earlier_layers:
    old_model_dir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/resnet_models/'
    old_model_name = 'pure_images_h512_w1024_more_labels_50epochs.pth'

run_title = str(resolution) + '.lr' + str(lr) + '.ss' + str(step_size) + '.bs' + str(batch_size) + 'mo' + str(momentum) + 'gamma' + str(gamma)
split = 0

#Potential num_splits_to_run = 1


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
        print('For test {}, split {}, Epoch {}/{}'.format(model_name, split, epoch, num_epochs - 1))
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
labels_df = pd.read_pickle(pid_df)
if multilabel_bool:
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

else:#Create a labels dictionary and pid list for the specific classes you're interested in:
    pids, labels, labels_dict = pre.createLabelsDict(labels_df, class_names)

#Generate the folds/partition
if kfold and new_folds: #Switch to gener
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

if kfold and not new_folds:

    kfold_file = open(kfold_filename, 'rb')
    kfold_inds = pickle.load(kfold_file)
    kfold_file.close()
    kfold_train_files = kfold_inds['trainfiles']
    kfold_test_files = kfold_inds['testfiles']

if not kfold and new_folds:
    partition, labels = pre.createTrainValSets(labels_df, class_names)
    with open(partition_filename, 'wb') as f:
        pickle.dump(partition, f)
    f.close()

    if equalize_classes:
        partition = pre.equalize_classes(class_names, partition, labels_df, ds_size)

if not kfold and not new_folds:
    partition_file = open(partition_filename, 'rb')
    partition = partition_file['partition']
    if equalize_classes:
        partition = pre.equalize_classes(class_names, partition, labels_df, ds_size)

if len(add_segment_states) > 0: #Switch to turn add new segmented images to the partition
    partition, labels_dict = pre.add_segmented_images(add_segment_states, partition, labels_dict, class_names)
    #Add segmented states, make the class number equal:
    no_class = pre.countClasses(labels_dict)
    min_imgs = np.int(np.min(no_class)*0.8)
    trainfiles = []
    for state in range(len(class_names)):
        inds_seg_class = np.where((['xcut' in filename for filename in list(labels_dict.keys())]) & (np.array(labels_dict.values())==state))[0]
        inds_full_class = np.where((['xcut' not in filename for filename in list(labels_dict.keys())]) & \
                                   (np.array(labels_dict.values()) == state) & ([filename not in partition['val'] for filename in list(labels_dict.keys())]))[0]

        # We want to fill the rest of the training with images that are not in the validation set, not segmented, and belong to the specific state
        # we also want the class number to be evenly distributed
        if len(inds_seg_class) > min_imgs:
            all_inds = np.concatenate((inds_seg_class, inds_full_class))
            import random
            random.shuffle(all_inds)
            for ind in all_inds[:min_imgs]:
                trainfiles.append(labels_dict.keys()[ind])

        if len(inds_seg_class) < min_imgs:
            for ind in inds_seg_class:
                trainfiles.append(labels_dict.keys()[ind])
            remainder_imgs = np.abs(len(inds_seg_class) - min_imgs)

            for ind in inds_full_class[:remainder_imgs]:
                trainfiles.append(labels_dict.keys()[ind])

    weights = [1/np.cbrt(np.min(no_class))] * len(class_names)
    class_weights = torch.FloatTensor(weights).cuda()


if tiled == True:
    import TiledArgusDS
    train_transform = transforms.Compose([transforms.Resize((int(resolution/2),int(resolution/2))),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomVerticalFlip(),
                                #transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize(mean,std),
                        ])


    test_transform = transforms.Compose([transforms.Resize((int(resolution/2), int(resolution/2))),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize(mean,std),
                                    ])


    train_ds = TiledArgusDS.TiledArgusTrainDS(basedir, partition['train'], labels_dict, resolution, nbn = nbn, transform = train_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

    val_ds = TiledArgusDS.TiledArgusTrainDS(basedir, partition['val'], labels_dict, resolution, nbn = nbn, transform = test_transform)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

else:
    import ArgusDS
    train_transform = transforms.Compose([#transforms.Resize((resolution, res2)),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Normalize(mean,std),
                        ])


    test_transform = transforms.Compose([transforms.Resize((resolution,res2)),
                                            transforms.ToTensor(),
                                            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize(mean,std),
                                    ])

    train_ds = ArgusDS.ArgusTrainDS(basedir, partition['train'], labels_dict, nbn = nbn, transform = train_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

    val_ds = ArgusDS.ArgusTrainDS(basedir, partition['val'], labels_dict, nbn = nbn, transform = test_transform)
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
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)

else:
    model_conv = models.vgg11()
    model_conv.classifier[6].out_features = 8
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)

model_conv = model_conv.to(device)
if multilabel_bool:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()


if train_earlier_layers == True:
    model_conv.load_state_dict(torch.load(old_model_dir + old_model_name))

    #This is to unfreeze earlier layers
    # Parameters of newly constructed modules have requires_grad=True by default
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    #for c in list(model_conv.children()):
    #    for p in c.parameters():
    #        p.requires_grad = True


# Decay LR by a factor of gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=no_epochs)
torch.save(model_conv.state_dict(), 'resnet_models/' + model_name + '.pth')

#model_conv.load_state_dict(torch.load('resnet_models/' + model_name + '.pth'))
conf_dt = post.calcConfusion(model_conv, dataloaders, class_names, device, mean, std, labels_df, waveparams, model_name, plotimgs  = False)
conf_dt.to_pickle(conf_folder + '/' + model_name + '.pickle')
print(no_class)
torch.cuda.empty_cache()
