import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable


class multi_target_resnet(torch.nn.Module):

    def __init__(self, model_core): #Initialize with the model (probably resnet50)
        super(multi_target_resnet, self).__init__()
        self.resnet_model = model_core

    def forward(self, x):
        x1 = self.resnet_model(x)
        loss =

