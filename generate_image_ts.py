from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as pl
import pickle
import pandas as pd
import os
from datetime import datetime as dt

from torch.autograd import Variable


######This will check the accuracy vs probability

#Configurations
pl.ion()
#Resolution for tiled ds, res1/res2 for nontiled
resolution = 512
mean = 0.48
std = 0.29 #This is from the 'calc mean'
modeltype = 'oblique'
class_names = ['B','C','D','E','F','G','Calm','NoVis']
no_class = [70, 172, 91, 57, 107, 105, 134, 60]
weights = [1/np.cbrt(no) for no in no_class]
class_weights = torch.FloatTensor(weights).cuda()
#This is the plot name of the accuracy figure

modelname = 'notiled_512'
modeltype = 'notiled' #tiled or no tiled
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
modelfolder = '/home/server/pi/homes/aellenso/Research/DeepBeach/resnet_models/ycseca/' + modelname + '/'
multilabel_bool = False
model_outfolder = '/home/server/pi/homes/aellenso/Research/DeepBeach/plots/' + modelname + '/'
if not os.path.exists(model_outfolder):
    os.mkdir(model_outfolder)


#labels_df = pd.DataFrame({'pid':list(prob_pid_dict.keys()), 'label':list(prob_pid_dict.values())})
###Info about the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load the model
model_conv = torchvision.models.resnet50(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_ftrs = model_conv.fc.in_features
nb_classes = len(class_names)
model_conv.fc = nn.Sequential(nn.Dropout(0.1),nn.Linear(num_ftrs, nb_classes))
model_conv = model_conv.to(device)



test_transform = transforms.Compose([transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize([mean, mean, mean],[std, std, std]),
                                ])

def test_model(model, dataloader, multilabel =False):

    model.eval()   # Set model to evaluate mode

    totalpreds = []
    totalprobs = []
    testpids = []
    testinps = []

    with torch.no_grad():
        for i, (inputs, pid) in enumerate(dataloader):
            torch.cuda.empty_cache()
            pid = pid[0]
            testpids.append(pid)
            testinps.append(inputs)
            inputs = inputs.to(device)
            try:
                outputs = model(inputs)
            except:
                continue
            if multilabel_bool:
                #This will return a multilabel prediction
                probs = torch.nn.functional.softmax(outputs)
                out_sigmoid = torch.sigmoid(outputs)
                t = Variable(torch.Tensor([0.5])).cuda()  # establish threshold
                preds = (out_sigmoid > t).float() * 1
                totalpreds.append(preds.cpu().numpy()[0])

            else:
                #This will return one prediction
                probs = torch.nn.functional.softmax(outputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()[0]
                totalpreds.append(preds)

            probs = probs[0].cpu().numpy()
            totalprobs.append(probs)

            if i % 100 == 0:
                print("On iteration {}".format(i))


            totalprobs_array = np.empty((len(totalprobs), len(probs)))
            for ti,probs in enumerate(totalprobs):
                totalprobs_array[ti,:] = probs

    return testpids, totalpreds, totalprobs_array



split_no = 0
models = os.listdir(modelfolder)
model = [mm for mm in models if 'foldno{}'.format(str(split_no)) in mm]
model = model[0]


test_IDs = os.listdir(basedir)

dates = []
for pid in test_IDs:
    dates.append(int(pid.split('.')[0]))
inds = np.argsort(dates)

test_IDs = [test_IDs[ii] for ii in inds]


import ArgusDS
test_ds = ArgusDS.ArgusTestDS(basedir, test_IDs[:120], transform = test_transform)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1) #num workers is how many subprocesses to use


model_conv.load_state_dict(torch.load(modelfolder + model))
model_conv = model_conv.to(device)

testpids, totalpreds, totalprobs_array = test_model(model_conv, test_dl, multilabel = multilabel_bool)

dates = []
for pid in testpids:
    dates.append(int(pid.split('.')[0]))
inds = np.argsort(dates)

allpreds = [totalpreds[ii] for ii in inds]
allprobs = totalprobs_array[inds,:]
allpids = [testpids[ii] for ii in inds]
time = [dt.fromtimestamp(date) for date in dates]

fig, ax = pl.subplots(1,1)
fig.set_size_inches(15,4)
img =ax.pcolor(time, np.arange(8), allprobs.T)
ax.plot(time, allpreds[:120], linewidth = 3, color = 'white')
ax.set_title(model)
ax.set_yticklabels(class_names)
ax.set_xlabel('Time (days)')
ax.set_ylabel('State')
c = pl.colorbar(img)
c.set_label('Class Weight')

pl.savefig(model_outfolder + 'BeachState_ts.png')



