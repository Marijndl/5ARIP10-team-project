import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CARNet
import numpy as np
import os
from spherical_coordinates import *

model = CARNet()
loss = nn.MSELoss()  # put loss function we have here
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust hyperparameters according to paper

#Load data
directory_2D = "D:\\CTA data\\Segments_deformed_2\\"
file_names_2D = os.listdir(directory_2D)
directory_3D = "D:\\CTA data\\Segments renamed\\"
file_names_3D = os.listdir(directory_3D)

data_2d = np.genfromtxt(os.path.join(directory_2D, file_names_2D[0]), delimiter=",")
data_3d = np.genfromtxt(os.path.join(directory_3D, file_names_3D[0]), delimiter=",")[1:, 1:4]

#Add row of zeros
data_2d = np.hstack((data_2d, np.zeros((data_2d.shape[0],1))))

#Normalize 3D
data_3d[:, 0] -= np.min(data_3d[:, 0])
data_3d[:, 1] -= np.min(data_3d[:, 1])
data_3d[:, 2] -= np.min(data_3d[:, 2])

#Convert to spherical:
origin_2D, spherical_2D = convert_to_spherical(data_2d)
origin_3D, spherical_3D = convert_to_spherical(data_3d)

origin_2D = torch.reshape(origin_2D, (3,1)).float()
origin_3D = torch.reshape(origin_3D, (3,1)).float()

spherical_2D = torch.reshape(spherical_2D, (3, 349)).float()
spherical_3D = torch.reshape(spherical_3D, (3, 349)).float()


ans = model.forward(origin_3D, spherical_3D, origin_2D, spherical_2D)

print(ans.shape)