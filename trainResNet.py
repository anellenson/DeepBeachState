from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
import copy
import pickle
from pytorchtools import EarlyStopping
import ArgusDS
import os


validate_only = False #Switch this if you want to train the model or just have it predict on the test dataset

#Input Data Information
#=================================
train_site = 'nbn'
class_names = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG']
trainfiles = 'labels/{}_daytimex_trainfiles.pickle'.format(train_site)
valfiles = 'labels/{}_daytimex_valfiles.pickle'.format(train_site)
testfiles = 'labels/{}_daytimex_testfiles.final.pickle'.format(train_site)
imgdir = '/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'
labels_dict = 'labels/{}_daytimex_labels_dict_five_aug.pickle'.format(train_site)
test_sites = ['nbn', 'duck'] #list of test sites to validate the model on

#Output and Model Info:
#==================================
prediction_fname = 'cnn_preds' #file to save prediction results
model_name = 'train_on_nbn_resnet512'
model_path = 'resnet_models/{}.pth'.format(model_name)
out_folder = 'model_output/{}'.format(model_name)
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

#Hyperparameters
#==================================
res_height = 512 #height
res_width = 512 #width
batch_size = 4
gray = True #This is a switch for grayscale or not
momentum = 0.9
gamma = 0.1 #amount to decay the learning rate
no_epochs = 1 #max number of epochs
step_size = 15 #when to decay the learning rate
lr = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(class_names)
criterion = nn.CrossEntropyLoss()


#Dataset Setup for pytorch loading.
#==================================
#Note that the test files are not fed into the dataloader for training, but are loaded below after training
with open(labels_dict, 'rb') as f:
    labels_dict = pickle.load(f)

with open(trainfiles, 'rb') as f:
    trainfiles = pickle.load(f)

with open(valfiles, 'rb') as f:
    valfiles = pickle.load(f)

print('Trainfiles length: {}'.format(len(trainfiles)))
print('Valfiles length: {}'.format(len(valfiles)))

transform = transforms.Compose([transforms.Resize((res_height, res_width)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ])

train_ds = ArgusDS.ArgusTrainDS(imgdir, trainfiles, labels_dict, transform = transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

val_ds = ArgusDS.ArgusTrainDS(imgdir, valfiles, labels_dict, transform = transform)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

dataloaders = {'train':train_dl, 'val':val_dl}
dataset_sizes = {'train':len(train_ds),'val':len(val_ds)}

#Model Initialization or Loading
#==================================================
#If "validate only" option is true, then the model will be loaded
model_conv = models.resnet50(pretrained = True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, nb_classes)

if validate_only == True:
    model_conv.load_state_dict(torch.load(model_path))
model_conv = model_conv.to(device)
optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=lr, momentum=momentum)  #Here to switch weights
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, 'min', factor=gamma, verbose=True, patience=8)
early_stopping = EarlyStopping()


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
        print('For train on {}, model {}, Epoch {}/{}'.format(train_site, model_name, epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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
                running_corrects += torch.sum(preds.float() == labels.data.float())

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                scheduler.step(epoch_loss)

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                if scheduler._last_lr[0] < 1E-3:
                    early_stopping(epoch_loss)


            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        if early_stopping.early_stop:
            print("Early Stopping")
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss, val_acc, train_acc, train_loss

def confusion_results(test_site):
    with open(testfiles, 'rb') as f:
        valfiles = pickle.load(f)

    test_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = False)

    CNN_preds = []
    truth = []
    id_list = []
    with torch.no_grad():
        for inputs, id, labels in test_dl:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype = torch.int64)

            outputs = model_conv(inputs)
            _, preds = torch.max(outputs,1)

            truth += list(labels.cpu().numpy())
            CNN_preds += list(preds.cpu().numpy())
            id_list += [ii for ii in id]

    return CNN_preds, truth, id_list

if not validate_only:
    #Training
    model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=no_epochs)
    torch.save(model_conv.state_dict(), model_path)

##Produce predictions for each site
CNN_results = {}
for test_site in test_sites:
    CNN_site_preds, truth_site, val_ids = confusion_results(test_site)
    CNN_results.update({'{}_CNN'.format(test_site):CNN_site_preds, '{}_truth'.format(test_site):truth_site, '{}_valfiles'.format(test_site):val_ids})

with open(out_folder + '{}.pickle'.format(prediction_fname), 'wb') as f:
    pickle.dump(CNN_results, f)

torch.cuda.empty_cache()
