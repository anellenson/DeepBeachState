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
import TiledArgusDS
import pickle
import preResnet as pre
import pickle
import postResnet as post
import plotTools
import numpy as np
import pandas as pd


def train_model(model, criterion, optimizer, scheduler, waveparams, num_epochs):
    since = time.time()
    val_loss = []
    val_acc = []

    train_loss = []
    train_acc = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

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
'''
First time through training
    - calculate the mean/std for pre-processing
    - partition the data between training and testing
'''



'''
#Calculate the mean/std
first_transform = transforms.Compose([transforms.Resize((res1, res2)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1))]) #Transform the data as if you were about to put it through a dataloader

#This will calculate the mean and standard deviation of each image in order to normalize the image. This only has to be done once for each dataset (e.g., oblique/rectified imagery)

mean,std = pre.calcMean(basedir, matfilename, first_transform, res1, res2) ##To do: calculate mean and variance for each combo oblique/rect, 1024, 526... etc. and create a lookup table
'''

totalres = 1024
lr = 0.01
batch_size = 4
type = 'oblique'
momentum = 0.9
gamma = 0.3
dropout = 0.1
waveparams = []
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
torch.cuda.empty_cache()
class_names = ['B','C','D','E','F','G','Calm','NoVis']
group_names = ['CalmB','CD','EF','G','SSZNoVis']
#Groups that we want each class to fall into
renamed_inds = {'NoBar':'HNoBarNoVis','B':'B','C':'CD','D':'CD','E':'EF','F':'EF','G':'G','H':'HNoBarNoVis','NoVis':'HNoBarNoVis'}
plot_dir  = '/home/server/pi/homes/aellenso/Research/DeepBeach/plots/withPybossa/'
no_epochs = 50
step_size = 15 #when to decay the learning rate
mean = 0.48
std = 0.29 #This is from the 'calc mean'
plot_fname_count = plot_dir + 'Tiled.New.resnetTrainInfo.count.x{0:3d}.y{1:3d}.lr{2:3.2e}.ss{3:2d}.bs{4:2d}.mo{5:3.2e}.gamma{6:3.2e}.png'.format(totalres,totalres,lr,step_size,batch_size,momentum,gamma)
plot_title = 'Tiled.' + str(totalres) + '.' + str(totalres) + '.lr' + str(lr) + '.ss' + str(step_size) + '.bs' + str(batch_size) + 'mo' + str(momentum) + 'gamma' + str(gamma)


#This loads the labels dictionary (for the ArgusDataset) and the labels dataframe (to create partitions) from the matfiles

labels_df = pd.read_pickle('/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/strlabel_df.pickle')

#partition, labels = pre.createTrainValSets(labels_df, class_names)
#p_pickle = open('partition_evennumber.pikl','wb')
#pickle.dump(partition,p_pickle)
#p_pickle.close()

#Reload old partition
p_pickle = open('../partition_comp_old.pikl','r')
partition = pickle.load(p_pickle)
p_pickle.close()

#Load new partitioned dataset based on whatever is not in the old dataset:
partition, labels = pre.createTrainSet(labels_df,partition, class_names)

'''
#Open the partition after the first time 
p_pickle = open('partition_noSSZ.pikl','r')
partition = pickle.load(p_pickle)
p_pickle.close()
'''

no_class = pre.countClasses(labels)
weights = [1/np.cbrt(no) for no in no_class]
class_weights = torch.FloatTensor(weights).cuda()


#####Only do that part once. Now load the partition each time
train_transform = transforms.Compose([transforms.Resize((int(totalres/2), int(totalres/2))),
                                        #transforms.RandomHorizontalFlip(),
                                        #transforms.RandomVerticalFlip(),
                                        #transforms.RandomRotation(40),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize([mean, mean, mean],[std, std, std]),
                                ])


test_transform = transforms.Compose([transforms.Resize((int(totalres/2), int(totalres/2))),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize([mean, mean, mean],[std, std, std]),
                                ])




#replace 'oblique' with 'rect' for data file names for the partition.
#Because the partitions are originally made with oblique imagery, the same images should be used but have to be called with their appropriate path


train_ds = TiledArgusDS.TiledArgusTrainDS(basedir, partition['train'], labels, totalres, transform = train_transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

val_ds = TiledArgusDS.TiledArgusTrainDS(basedir, partition['val'], labels, totalres, transform = test_transform)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

dataloaders = {'train':train_dl, 'val':val_dl}

dataset_sizes = {'train':len(train_ds),'val':len(val_ds)}

#Set up the model, load a new one or decide to load a pretrained one

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(class_names)
model_conv = models.resnet50(pretrained = True)
num_ftrs = model_conv.fc.in_features
nb_classes = len(class_names)
model_conv.fc = nn.Sequential(nn.Dropout(dropout),nn.Linear(num_ftrs, nb_classes))
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss(class_weights)



#model_conv.load_state_dict(torch.load('models/resnet50/resnet50.comp_old.512.512.lr0.01.ss15.bs12mo0.9gamma0.3.pth'))

#This is to unfreeze earlier layers
#for c in list(model_conv.children()):
#    print(c)
#    for p in c.parameters():
#        p.requires_grad = True


# Parameters of newly constructed modules have requires_grad=True by default
# Observe that only parameters of final layer are being optimized as
# opposed to before.

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)

# Decay LR by a factor of 0.1 every 50 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, waveparams, num_epochs=no_epochs)
#post.CAM(model_conv, dataloaders, device, class_names, mean, std, cam_dir)

#Save model and other info about this run
#torch.save(model_conv.state_dict(), 'models/resnet50/resnet50.tiled_notransform.' + plot_title + '.pth')

#Call the post processing routine - plot the training/loss curves and the confusion table. Also plot the grouped confusion table
conf_dt = post.calcConfusion(model_conv, dataloaders, class_names, device, mean, std, labels_df, waveparams, plotimgs = False)
#confpercent_dt = post.calcConfusionPercent(model_conv,dataloaders,class_names, labels_df, device)
#confusion_groups = post.calcConfusedGroups(conf_dt,renamed_inds)
plotTools.trainInfo(conf_dt, class_names, val_acc, train_acc, val_loss, train_loss, plot_fname_count, plot_title)
#plotTools.confusionTable(conf_dt, class_names, plot_fname_count, plot_title)

#plotTools.confusionTable(confusion_groups,[cc for cc in confusion_groups.columns], grouped_confname, plot_title)

#conf_dt.to_pickle('../' + plot_title + '.pikl')
