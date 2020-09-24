from __future__ import print_function, division

import os
from PIL import Image
import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from grad_cam import BackPropagation, GuidedBackPropagation, GradCAM
import pickle
import cv2
from collections import Counter
import argparse


parser = argparse.ArgumentParser(description = 'specify which model')
parser.add_argument('-m', '--modelname')
parser.add_argument('-state', '--beachstate')
parser.add_argument('-ii', '--start_index', type = int)
parser.add_argument('-testsite', '--testsite')
parser.add_argument('-trainsite', '--trainsite')
parser.add_argument('-imgdir', '--imgdirectory')
parser.add_argument('-outdir', '--outdirectory')
parser.add_argument('-topk', '--topk')
args = parser.parse_args()
modelname = args.modelname
beachstate = args.beachstate
ii = args.start_index
testsite = args.testsite
trainsite = args.trainsite
imgdir = args.imgdir
out_folder = args.outdir
topk = args.topk

def load_images(test_IDs, res_height, res_width):
    images = []
    raw_images = []

    for ID in test_IDs:
        image, raw_image = preprocess(ID, res_height, res_width)
        images.append(image)
        raw_images.append(raw_image)

    return images, raw_images

def preprocess(image_path, res_height, res_width):
    transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])
                                       #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert("RGB")
        raw_image = transform(image)

    return raw_image, raw_image

def gradient_im(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0

    return gradient.astype(int)


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.numpy().transpose(1,2,0)) / 2

    cv2.imwrite(filename, np.uint8(gcam))


torch.cuda.empty_cache()
modelpath = 'resnet_models/train_on_{}/{}.pth'.format(trainsite, modelname)
res_height = 512 #height
res_width = 512 #width
classes = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']

##load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(classes)
model_conv = models.resnet50()
target_layer = 'layer4'
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, nb_classes) # check is there really drop out

model_conv.load_state_dict(torch.load(modelpath))
model_conv = model_conv.to(device)
model_conv.eval()

with open('labels/{}_labels_dict_five_aug.pickle'.format(testsite), 'rb') as f:
    labels_dict = pickle.load(f)

#filter so you get each class
test_labels= np.array([labels_dict[pid] for pid in valfiles])
test_IDs = []
class_pids = []
beachstatenum = classes.index(beachstate)
class_pids = np.where(test_labels == beachstatenum)[0]
test_IDs += [valfiles[cc] for cc in class_pids[ii:ii+5]] #Had to limit to only 5 images before I ran out of memory generating the grad cam visualizations
test_IDs = [imgdir + '/'+tt for tt in test_IDs]


images, raw_images = load_images(test_IDs, res_height, res_width)
images = torch.stack(images).to(device)


guided_gradcams_dict = {}
for j, (image, img_ID) in enumerate(zip(images, test_IDs)):

    image = image.unsqueeze(dim = 0)
    ID = img_ID.split('/')[-1]
    ID = ID.split('.')[0]
    if testsite == 'nbn':
        ID = ID.split('_')[1]

    bp = BackPropagation(model=model_conv)
    probs, ids = bp.forward(image)#generate the top predictions

    gcam = GradCAM(model=model_conv)
    _ = gcam.forward(image)

    gbp = GuidedBackPropagation(model=model_conv)
    _ = gbp.forward(image)

    for i in range(topk):

        torch.cuda.empty_cache()

        prediction = classes[ids[0, i]]

        #for ax in [ax_cam, ax_gbp, ax_gbpcam]:
        ax[j,i+1].set_title(prediction, fontsize = 22)
        ax[j,i+1].imshow(image.detach().squeeze().cpu().numpy().transpose(1,2,0))
        ax[j,i+1].axis('off')

        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        #print("\t#{}: {} ({:.5f})".format(ii + j, classes[ids[0, i]], ensemble_probs[prediction]))
        print("\t#{}: {} ({:.5f})".format(ii + j, classes[ids[0, i]], probs[0,i]))

        # Guided Backpropagation
        gbp_gradient = gradient_im(gradients[0])
        ax.imshow(gbp_gradient)
        ax.set_title('{0} {1:.2f}'.format(prediction, probs[0,i]))
        ax.set_title('{0}'.format(prediction))

        #Grad-Cam
        grad_cam = regions[0,0].cpu().numpy()

        # Guided Grad-CAM
        guided_gradcam = gradient_im(torch.mul(regions, gradients)[0])
        guided_gradcam = guided_gradcam[:,:,0]
        #find the mode and zero it
        def clean_gradcam(guided_gradcam):
            counts = Counter(list(guided_gradcam.flatten()))
            max_count = np.max(counts.values())
            mode = [k for k,v in counts.items() if v == max_count][0]
            guided_gradcam = guided_gradcam-mode
            return guided_gradcam


        ax[j, i+1].imshow(guided_gradcam, alpha = 0.5, cmap = 'hot', vmin = 0, vmax = 100)
        ax[j,i+1].set_title('{0}'.format(prediction))

        guided_gradcams_dict.update({img_ID:(guided_gradcam, prediction)})


with open(out_dir + '/{}_grad_cam_dict_{}.pickle'.format(beachstate, ii), 'wb') as f:
    pickle.dump(guided_gradcams_dict, f)

fig.savefig(plots_dir + '/GradCam_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')
