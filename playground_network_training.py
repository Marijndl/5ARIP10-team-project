import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CARNet
import numpy as np
import os
from spherical_coordinates import *
from torchsummary import summary

def mPD_loss(registered, original):
    loss = torch.mean(torch.sum(torch.abs(registered - original), dim=1))
    return loss

model = CARNet()
loss = nn.MSELoss()  # put loss function we have here
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adjust hyperparameters according to paper

#Load data
directory_2D = ".\\CTA data\\Segments_deformed_2\\"
file_names_2D = os.listdir(directory_2D)
directory_3D = ".\\CTA data\\Segments renamed\\"
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

origin_2D = torch.reshape(origin_2D, (1,3,1)).float()
origin_3D = torch.reshape(origin_3D, (1,3,1)).float()

spherical_2D = torch.unsqueeze(torch.transpose(spherical_2D, 0, 1), dim=0).float()
spherical_3D = torch.unsqueeze(torch.transpose(spherical_3D, 0, 1), dim=0).float()

# model.cuda()
# origin_2D.cuda()
# origin_3D.cuda()
# spherical_2D.cuda()
# spherical_3D.cuda()

ans = model.forward(origin_3D, spherical_3D, origin_2D, spherical_2D)
# print(summary(model, (origin_3D, spherical_3D, origin_2D, spherical_2D)))

# deformed = torch.tensor([])
# original = torch.tensor([])
#
# #Add deformation to 3D line
# spherical_3D[:,1:,:] += ans[0]
#
# for idx in range(ans.shape[0]):
#     #Convert back to cartesian
#     cartesian_2D = convert_back(torch.transpose(origin_2D[idx], 0, 1).detach().numpy(), torch.transpose(spherical_2D[idx], 0, 1).detach().numpy()).squeeze()
#     cartesian_3D = convert_back(torch.transpose(origin_3D[idx], 0, 1).detach().numpy(), torch.transpose(spherical_3D[idx], 0, 1).detach().numpy()).squeeze()
#
#     #Project to 2D
#     cartesian_3D[:,2] = np.zeros(cartesian_3D.shape[0])
#
#     original = torch.cat((original, torch.transpose(torch.from_numpy(cartesian_2D).float(), 0, 1)), dim=0)
#     deformed = torch.cat((deformed, torch.transpose(torch.from_numpy(cartesian_3D).float(), 0, 1)), dim=0)
#
# deformed = deformed.unsqueeze(dim=0)
# original = original.unsqueeze(dim=0)
# def mPD_loss(registered, original):
#     loss = torch.sum(torch.mean(torch.sum(torch.abs(registered - original), dim=1), dim=1))
#     return loss
#
# loss = mPD_loss(deformed, original)
#
# print(ans.shape)