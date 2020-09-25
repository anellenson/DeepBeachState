from __future__ import print_function, division

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from utils.grad_cam import BackPropagation, GuidedBackPropagation, GradCAM
import pickle
import cv2
import argparse

parser = argparse.ArgumentParser(description = '''
                                               
This script generates Guided-GradCAM images. 
Script adapted from:
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26


        Directory Information
        ===================================
        Note that the outdir will automatically be: model_output/model_name/
        -mpath:         modelpath, including the model name, that yu want to make the visualization for
        -imgpath:      image path to evaluate
        
        CNN Options
        ====================================
        -topk:          number of choices to visualize - i.e., the first choice, second choice, third choice 
                        that the CNN would make as output by the softmax function
        
        
 
''')
parser.add_argument('-m', '--modelpath')
parser.add_argument('-i', '--imgpath')
parser.add_argument('-t', '--topk')

args = parser.parse_args()
modelpath = args.modelpath

topk = int(args.topk)
imgpath = args.imgpath

pid = imgpath.split('/')[-1]
modelname = modelpath.split('/')[-1][:-4]

def preprocess(image_path, res_height, res_width):
    transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])

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

image, raw_image = preprocess(imgpath, res_height, res_width)
image = image.unsqueeze(dim = 0).cuda()

ggcam_dict = {}

bp = BackPropagation(model=model_conv)
probs, ids = bp.forward(image)#generate the top predictions

gcam = GradCAM(model=model_conv)
_ = gcam.forward(image)

gbp = GuidedBackPropagation(model=model_conv)
_ = gbp.forward(image)

for i in range(topk):

    torch.cuda.empty_cache()

    # Guided Backpropagation
    gbp.backward(ids=ids[:, [i]])
    gradients = gbp.generate()

    # Grad-CAM
    gcam.backward(ids=ids[:, [i]])
    regions = gcam.generate(target_layer=target_layer)

    print("\t#{}: {} ({:.5f})".format(i, classes[ids[0, i]], probs[0,i]))

    # Guided Backpropagation
    #Grad-Cam
    grad_cam = regions[0,0].cpu().numpy()

    # Guided Grad-CAM
    guided_gradcam = gradient_im(torch.mul(regions, gradients)[0])
    guided_gradcam = guided_gradcam[:,:,0]

    ggcam_dict.update({'{}_choice_ggcam'.format(i):grad_cam})

#Save out the predictions and probabilities for each prediction
probs = probs.detach().cpu().numpy().squeeze()
ids = ids.detach().cpu().numpy().squeeze()
ggcam_dict.update({'ids':ids, 'probs':probs})

with open('model_output/{}/ggcam_{}.pickle'.format(modelname, pid[:-4]), 'wb') as f:
    pickle.dump(ggcam_dict, f)

