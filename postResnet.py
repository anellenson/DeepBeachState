import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def imshow(inp, mean, std, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean, mean, mean])
    std = np.array([std, std, std])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def calcConfusion(model_conv, dataloaders, class_names, device, mean, std, labels_df, waveparams, plotimgs  = False):
    #Set plot images to "True" if you want to plot the confused images.
    nb_classes = len(class_names)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    confused_dir = '/home/server/pi/homes/aellenso/Research/DeepBeach/plots/confused/'
    #First remove everything in the directory
#confusion_grouped_matrix = torch.zeros(length(group_list), length(group_list))
    if plotimgs:
        for cc in class_names:
            filelist = os.listdir(confused_dir + cc)
            for ff in filelist:
                os.remove(os.path.join(confused_dir,cc,ff))

    cnt = 0
    with torch.no_grad():
        for i, (inputs, id, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            if waveparams != []:

                wavedata = []
                for ii in id:
                    wavevalues = []
                    for ww in waveparams:
                        value = labels_df[(labels_df['file'] == ii)][ww].values[0]
                        wavevalues.append(value)
                    wavedata.append(wavevalues)
                wavedata = torch.tensor(wavedata).type('torch.FloatTensor').to(device)
                outputs = model_conv(inputs, wavedata)
                probabilities = torch.nn.softmax(output)
                #top 3 probabilities


            if waveparams == []:
                outputs = model_conv(inputs)
            _, preds = torch.max(outputs, 1)

            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            if plotimgs:
                for pi,pp in enumerate(preds):
                    fig = plt.figure(1)
                    plt.clf()
                    imshow(inputs[pi].cpu(), mean,std)
                    plt.title('Predicted {} and Truth {}'.format(class_names[pp],class_names[classes[pi]]))
                    #Text here for the top probabilities
                    #plt.pause(0.5)
                    plt.savefig(confused_dir + '{}/img{}.png'.format(class_names[classes[pi]],cnt), dpi = 800)
                    cnt +=1

    conf_dt = pd.DataFrame(columns = class_names, index = class_names, data = confusion_matrix.numpy())
    conf_dt = conf_dt

    return conf_dt

def calcConfusionPercent(model_conv, dataloaders, class_names, labels_df, waveparams, device):
    #Set plot images to "True" if you want to plot the confused images.
    nb_classes = len(class_names)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    confused_dir = '/home/server/pi/homes/aellenso/Research/DeepBeach/plots/confused/'
    #First remove everything in the directory
#confusion_grouped_matrix = torch.zeros(length(group_list), length(group_list))
    cnt = 0
    with torch.no_grad():
        for i, (inputs, id, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            if waveparams != []:

                wavedata = []
                for ii in id:
                    wavevalues = []
                    for ww in waveparams:
                        value = labels_df[(labels_df['file'] == ii)][ww].values[0]
                        wavevalues.append(value)
                    wavedata.append(wavevalues)
                wavedata = torch.tensor(wavedata).type('torch.FloatTensor').to(device)
                outputs = model_conv(inputs, wavedata)


            if waveparams == []:
                outputs = model_conv(inputs)

            lossvals = outputs.cpu().numpy()

            for ti,t in enumerate(classes.view(-1)):
                losspercent = np.zeros(len(lossvals[ti,:]))
                nonzeroloss = np.where(lossvals[ti,:] > 0)[0]
                losspercent[nonzeroloss] = lossvals[ti,nonzeroloss]/np.sum(lossvals[ti,nonzeroloss])
                confusion_matrix[t.long(), :] += torch.Tensor(losspercent)

    conf_dt = pd.DataFrame(columns = class_names, index = class_names, data = confusion_matrix.numpy())
    conf_dt = conf_dt

    return conf_dt

def calcConfusedGroups(conf_dt, renamed_inds):
    conf_dtgroup = conf_dt.rename(index = renamed_inds)
    conf_dtgroup = conf_dtgroup.rename(columns = renamed_inds)
    conf_dtgroup = conf_dtgroup.groupby(conf_dtgroup.columns, axis = 1).sum().groupby(conf_dtgroup.index, axis =0).sum()
    return conf_dtgroup

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, id, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            wavedata = []
            for ii in id:
                wavevalues = []
                for ww in waveparams:
                    value = labels_df[(labels_df['file'] == ii)][ww].values[0]
                    wavevalues.append(value)
                wavedata.append(wavevalues)

            wavedata = torch.tensor(wavedata).type('torch.FloatTensor').to(device)
            outputs = model(inputs,wavedata)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def CAM(model,dataloaders, device, class_names, mean, std, cam_dir):
    model.eval()
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get('layer4').register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (512, 512)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for bb in np.arange(bz):
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv[bb].reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


    with torch.no_grad():
        for i, (inputs, id, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            CAMs = returnCAM(features_blobs[0], weight_softmax, preds.cpu().numpy())
            cnt = 0
            for pi,pp in enumerate(preds):
                fig = plt.figure(1)
                plt.clf()
                imshow(inputs[pi].cpu(), mean,std)
                plt.imshow(CAMs[pi], alpha = 0.4)
                plt.colorbar()
                plt.title('Predicted {} and Truth {}'.format(class_names[pp],class_names[classes[pi]]))
                #plt.pause(0.5)
                plt.savefig(cam_dir + '{}/img{}.png'.format(class_names[preds[pi]],cnt), dpi = 800)
                cnt +=1

