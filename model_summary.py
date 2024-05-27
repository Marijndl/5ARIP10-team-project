import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CARNet
import numpy as np
import os
from spherical_coordinates import *
from torch.autograd import Variable
from tqdm import tqdm
import warnings
from data_loaders import *
from torchsummary import summary


model = CARNet()

data_3D = torch.load('.\\CTA data\\shape_3D.pt')

# print(torch.sum(torch.isnan(data_3D)))
print(summary(model, input_size=[(3,1), (3,349), (3,1), (3,349)], batch_size=512))