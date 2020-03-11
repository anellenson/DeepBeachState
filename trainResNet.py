from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle
from torch.autograd import Variable
from pytorchtools import EarlyStopping




###Data info

#load the labels dataframe

pid_df_dict = {'train_on_duck':'labels/duck_daytimex_labels_df.pickle', 'train_on_nbn':'nbn_daytimex_labels.pickle'} ############no longer need this, since we're doing it all with dictionaries
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
res_height = 512 #height
res_width = 512 #width
batch_size = 4

gray = True #This is a switch for grayscale or not
momentum = 0.9
gamma = 0.1
equalize_classes = True
no_epochs = 120
step_size = 15 #when to decay the learning rate
mean = duck_gray_mean
std = duck_gray_std
waveparams = []
#pretrained = True
multilabel_bool = False
pretrained = False
train_earlier_layers = False
CNNtype = 'resnet'
for train_site in ['nbn', 'nbn_duck']:
    for augtype in ['nbn_to_duck_aug']:
        for runno in range(10):
            lr = 0.01

            ##saveout info
            model_name = 'resnet512_{}_{}'.format(augtype, runno)



            basedirs = ['/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
                        '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/']

            out_folder = 'model_output/train_on_{}/{}/'.format(train_site,  model_name)

            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            def load_train_and_valfiles(train_site):

                valfiles = 'labels/{}_daytimex_valfiles.{}_imgs.pickle'.format(train_site, augtype)
                trainfiles = 'labels/{}_daytimex_trainfiles.{}_imgs.pickle'.format(train_site, augtype)

                with open(valfiles, 'rb') as f:
                    valfile = pickle.load(f)

                with open(trainfiles, 'rb') as f:
                     trainfile = pickle.load(f)

                return valfile, trainfile

            valfiles = []
            trainfiles = []
            train_site_list = train_site.split('_')
            for tt in train_site_list:
                valfile, trainfile = load_train_and_valfiles(tt)
                valfiles += valfile
                trainfiles += trainfile

            print('Trainfiles length: {}'.format(len(trainfiles)))
            print('Valfiles length: {}'.format(len(valfiles)))

            with open('labels/duck_labels_dict_{}.pickle'.format(augtype), 'rb') as f:
                labels_dict = pickle.load(f)

            with open('labels/nbn_labels_dict_{}.pickle'.format(augtype), 'rb') as f:
                nbn_dict = pickle.load(f)

            labels_dict.update(nbn_dict)
            print('labels dictionary length: {}'.format(len(list(labels_dict.keys()))))


            ######################################################################################################################

            #This loads the labels dictionary (for the ArgusDataset) and the labels dataframe (to create partitions) from the matfiles


            import ArgusDS
            train_transform = transforms.Compose([transforms.Resize((res_height, res_width)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        # transforms.Normalize(mean,std),
                                ])


            test_transform = transforms.Compose([transforms.Resize((res_height,res_width)),
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                    # transforms.Normalize(mean,std),
                                            ])

            train_ds = ArgusDS.ArgusTrainDS(basedirs, trainfiles, labels_dict, transform = train_transform)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True) #change batch size?

            val_ds = ArgusDS.ArgusTrainDS(basedirs, valfiles, labels_dict, transform = test_transform)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

            dataloaders = {'train':train_dl, 'val':val_dl}

            dataset_sizes = {'train':len(train_ds),'val':len(val_ds)}

            #Set up the model, load a new one or decide to load a pretrained one

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            nb_classes = len(class_names)


            if pretrained == True:
                if CNNtype == 'resnet':
                    model_conv = models.resnet50(pretrained = True)

                if CNNtype == 'inception_resnet':
                    from pretrainedmodels import inceptionresnetv2
                    model_conv = inceptionresnetv2(pretrained=True, num_classes = nb_classes)


            if pretrained == False:
                if CNNtype == 'resnet':
                    model_conv = models.resnet50()

                if CNNtype == 'inception':
                    model_conv = models.inception_v3()

                if CNNtype == 'inception_resnet':
                    from pretrainedmodels import inceptionresnetv2
                    model_conv = inceptionresnetv2(pretrained=None, num_classes = nb_classes)

                if CNNtype == 'mobilenet':
                    model_conv = models.mobilenet_v2()

                if CNNtype == 'alexnet':
                    model_conv = models.alexnet()


            if CNNtype == 'resnet':
                num_ftrs = model_conv.fc.in_features
                model_conv.fc = nn.Linear(num_ftrs, nb_classes)

            if CNNtype == 'mobilenet':
                model_conv.classifier[1].out_features = nb_classes

            if CNNtype == 'alexnet':
                model_conv.classifier[6].out_features = nb_classes


            if train_earlier_layers == True:
                model_conv.load_state_dict(torch.load(old_model))
                #This is to unfreeze earlier layers
                # Parameters of newly constructed modules have requires_grad=True by default
                # Observe that only parameters of final layer are being optimized as
                # opposed to before.
                c = list(model_conv.parameters())[:-3]
                for p in c:
                    p.requires_grad = False


            model_conv = model_conv.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=lr, momentum=momentum)
            # Decay LR by a factor of gamma every step_size epochs
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, 'min', factor=gamma, verbose=True, patience=8)

            early_stopping = EarlyStopping()
            ##################################################################################################################################################################
            ######################################################################################################################
            ######################################################################################################################

            def train_model(model, criterion, optimizer, scheduler, CNNtype, num_epochs):
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
                        if phase == 'val':
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
                with open('labels/{}_daytimex_valfiles.noaug_imgs.pickle'.format(test_site)) as f:
                    valfiles = pickle.load(f)

                val_ds = ArgusDS.ArgusTrainDS(basedirs, valfiles, labels_dict, transform = test_transform)
                val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = True)

                CNN_preds = []
                truth = []

                with torch.no_grad():
                    for inputs, id, labels in val_dl:
                        inputs = inputs.to(device)
                        labels = labels.to(device, dtype = torch.int64)

                        outputs = model_conv(inputs)
                        _, preds = torch.max(outputs,1)

                        truth += list(labels.cpu().numpy())
                        CNN_preds += list(preds.cpu().numpy())

                return CNN_preds, truth


            model_conv, val_loss, val_acc, train_acc, train_loss = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, CNNtype, num_epochs=no_epochs)
            torch.save(model_conv.state_dict(), 'resnet_models/train_on_{}/{}.pth'.format(train_site, model_name))


            #Save out train info
            train_info_dict = {'val_loss':val_loss, 'val_acc':val_acc, 'train_acc':train_acc, 'train_loss':train_loss}
            with open(out_folder + 'train_specs.pickle'.format(train_site, model_name), 'wb') as f:
                pickle.dump(train_info_dict, f)

            # #Produce predictions for each site

            CNN_results = {}
            for test_site in ['duck', 'nbn']:
                CNN_site_preds, truth_site = confusion_results(test_site)

                CNN_results.update({'{}_CNN'.format(test_site):CNN_site_preds, '{}_truth'.format(test_site):truth_site})

            with open(out_folder + 'cnn_preds.pickle', 'wb') as f:
                pickle.dump(CNN_results, f)


            specs = ['Resolution: h{} x w{}'.format(res_height, res_width), 'num_epochs: {}'.format(no_epochs), 'batch_size: {}'.format(batch_size), 'learning_rate: {}'.format(lr)]

            with open(out_folder + 'model_info.txt', 'wb') as f:
                for spec in specs:
                    f.writelines(spec + '\n')



            torch.cuda.empty_cache()
