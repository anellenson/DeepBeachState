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
import ArgusDS
import pickle
import preResnet as pre
import pickle
import postResnet as post
import plotTools
import numpy as np



class waveCNN(nn.Module):

    def __init__(self, waveparams, nb_classes):
        super(waveCNN, self).__init__()
        self.cnn = models.resnet50(pretrained = True)
        self.cnn.fc = nn.Sequential(nn.Dropout(dropout),nn.Linear(self.cnn.fc.in_features, 2048))
        self.fc1 = nn.Linear(2048 + len(waveparams), nb_classes)
        #self.fc2 = nn.Linear(2048, nb_classes)

    def forward(self, image, wavedata):
        x1 = self.cnn(image)
        if len(wavedata.shape) == 1:
            x = torch.cat((x1, wavedata.unsqueeze(1)), dim = 1)
        if len(wavedata.shape) > 1:
            x = torch.cat((x1, wavedata), dim = 1)
        x = self.fc1(x)
        #x = torch.nn.functional.relu(self.fc1(x))
        #x = self.fc2(x)

        return x


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

                wavedata = []
                for ii in id:
                    wavevalues = []
                    for ww in waveparams:
                        value = labels_df[(labels_df['file'] == ii)][ww].values[0]
                        wavevalues.append(value)
                    wavedata.append(wavevalues)

                # zero the parameter gradients
                wavedata = torch.tensor(wavedata).type('torch.FloatTensor').to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs,wavedata)
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


type = 'oblique'
class_names = ['B','C','D','E','F','G','H','NoBar','NoVis']


if type == 'rect':
    basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/c0/test/'
    #Load the matfile with the labels (vector), labelled files (cell array of strings), and a vector with a 1 or 0 indicating if the image was duplicated for training
    matfilename = '/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/labeled_files_final_rect.mat'

if type == 'oblique':
    basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
    matfilename = '/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/labeled_files_final_oblique.mat'

#This loads the labels dictionary (for the ArgusDataset) and the labels dataframe (to create partitions) from the matfiles
#labels, labels_df = pre.loadLabels(matfilename, waveparams)
#partition = pre.createTrainValSets(labels_df, class_names)

'''
#Calculate the mean/std
first_transform = transforms.Compose([transforms.Resize((res1, res2)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1))]) #Transform the data as if you were about to put it through a dataloader

#This will calculate the mean and standard deviation of each image in order to normalize the image. This only has to be done once for each dataset (e.g., oblique/rectified imagery)

mean,std = pre.calcMean(basedir, matfilename, first_transform, res1, res2) ##To do: calculate mean and variance for each combo oblique/rect, 1024, 526... etc. and create a lookup table
'''


res1 = 512
res2 = 512
lr = 0.01
batch_size = 8
type = 'oblique'
momentum = 0.9
gamma = 0.1
dropout = 0.1
waveparams = ['Hs']

torch.cuda.empty_cache()
class_names = ['B','C','D','E','F','G','H','NoBar','NoVis']
no_class = [53, 90, 56, 36, 57, 59, 37, 108, 55]
#no_class = [53, 76, 23, 28, 41, 48, 36, 92, 45]
weights = [1/np.cbrt(no) for no in no_class]
class_weights = torch.FloatTensor(weights).cuda()
group_names = ['NoBarB','CD','EF','G','HNoVis']
#Groups that we want each class to fall into
renamed_inds = {'NoBar':'HNoBarNoVis','B':'B','C':'CD','D':'CD','E':'EF','F':'EF','G':'G','H':'HNoBarNoVis','NoVis':'HNoBarNoVis'}
plot_dir  = '/home/server/pi/homes/aellenso/Research/DeepBeach/plots/resnet/withWaves/'
no_epochs = 30
step_size = 12 #when to decay the learning rate
for no in np.arange(6,25):
    plot_fname_percent = plot_dir + 'resnetTrainInfo.Hs' + str(no) + '.' + type + '.x{0:3d}.y{1:3d}.lr{2:3.2e}.ss{3:2d}.bs{4:2d}.mo{5:3.2e}.gamma{6:3.2e}.png'.format(res1,res2,lr,step_size,batch_size,momentum,gamma)
    plot_fname_count = plot_dir + 'resnetTrainInfo.Hs' + str(no) + '.' + type  + '.x{0:3d}.y{1:3d}.lr{2:3.2e}.ss{3:2d}.bs{4:2d}.mo{5:3.2e}.gamma{6:3.2e}.png'.format(res1,res2,lr,step_size,batch_size,momentum,gamma)
    grouped_confname = plot_dir + 'resnetGroupedMatrix.Hs' + str(no) + '.' + type + '.x{0:3d}.y{1:3d}.lr{2:3.2e}.ss{3:2d}.bs{4:2d}.mo{5:3.2e}.gamma{6:3.2e}.png'.format(res1,res2,lr,step_size,batch_size,momentum,gamma)
    plot_title = 'Hs' + str(no) + '.' + type + str(res1) + '.' + str(res2) + '.lr' + str(lr) + '.ss' + str(step_size) + '.bs' + str(batch_size) + 'mo' + str(momentum) + 'gamma' + str(gamma)


    print("On run " + plot_title[:-3])

    mean = 0.48
    std = 0.29 #This is from the 'calc mean'

    if type == 'rect':
        basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/c0/test/'
        #Load the matfile with the labels (vector), labelled files (cell array of strings), and a vector with a 1 or 0 indicating if the image was duplicated for training
        matfilename = '/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/labeled_files_final_rect.mat'

    if type == 'oblique':
        basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
        matfilename = '/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/labeled_files_final_oblique.mat'

    #This loads the labels dictionary (for the ArgusDataset) and the labels dataframe (to create partitions) from the matfiles
    labels, labels_df = pre.loadLabels(matfilename,waveparams)
    #partition = pre.createTrainValSets(labels_df, class_names)
    #p_pickle = open('partition.pikl','wb')
    #partition = pickle.dump(partition,p_pickle)
    #p_pickle.close()

    #####Only do that part once. Now load the partition each time
    train_transform = transforms.Compose([transforms.Resize((res2,res1)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(40),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize([mean, mean, mean],[std, std, std]),
                                    ])


    test_transform = transforms.Compose([transforms.Resize((res2,res1)),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize([mean, mean, mean],[std, std, std]),
                                    ])


    #Open the partition on the subsequent
    p_pickle = open('partition.pikl','r')
    partition = pickle.load(p_pickle)
    p_pickle.close()

    #replace 'oblique' with 'rect' for data file names for the partition.
    #Because the partitions are originally made with oblique imagery, the same images should be used but have to be called with their appropriate path
    if 'rect' in matfilename:
        partition = pre.renamePartition(partition)


    train_ds = ArgusDS.ArgusTrainDS(basedir, partition['train'], labels, transform = train_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = 4) #change batch size?

    val_ds = ArgusDS.ArgusTrainDS(basedir, partition['val'], labels, transform = test_transform)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = 4)

    dataloaders = {'train':train_dl, 'val':val_dl}

    dataset_sizes = {'train':len(train_ds),'val':len(val_ds)}

    #Set up the model, load a new one or decide to load a pretrained one

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_classes = len(class_names)
    model_conv = waveCNN(waveparams, nb_classes)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss(class_weights)



    #model_conv.load_state_dict(torch.load('models/resnet50/resnet50.oblique512.512.lr0.01.ss20.bs16mo0.9gamma0.6.pth'))

    #This is to unfreeze earlier layers
    #for c in list(model_conv.children()):
    #    print(c)
    #    for p in c.parameters():
    #        p.requires_grad = True


    # Parameters of newly constructed modules have requires_grad=True by default
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.

    optimizer_conv = optim.SGD(model_conv.fc1.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 50 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

    model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, waveparams, num_epochs=no_epochs)
    #post.CAM(model_conv, dataloaders, criterion)

    #Save model and other info about this run
    #torch.save(model_conv.state_dict(), 'models/resnet50/resnet50.' + plot_title + '.pth')

    #Call the post processing routine - plot the training/loss curves and the confusion table. Also plot the grouped confusion table
    conf_dt = post.calcConfusion(model_conv, dataloaders, class_names, device, mean, std, labels_df, waveparams, plotimgs = False)
    #confpercent_dt = post.calcConfusionPercent(model_conv,dataloaders,class_names, labels_df, device)
    #confusion_groups = post.calcConfusedGroups(conf_dt,renamed_inds)
    #plotTools.trainInfo(conf_dt, class_names, val_acc, train_acc, val_loss, train_loss, plot_fname_count, plot_title)
    #plotTools.confusionTable(conf_dt, class_names, plot_fname_count, plot_title)

    #plotTools.confusionTable(confusion_groups,[cc for cc in confusion_groups.columns], grouped_confname, plot_title)

    conf_dt.to_pickle('../results/withWaves/Hs/' + plot_title + '.pikl')
