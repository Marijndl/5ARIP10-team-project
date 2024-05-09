import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CARNet
import numpy as np
import os
from spherical_coordinates import *


#Load data
directory_2D = "D:\\CTA data\\Segments_deformed_4\\"
file_names_2D = os.listdir(directory_2D)
directory_3D = "D:\\CTA data\\Segments bspline 353\\"
file_names_3D = os.listdir(directory_3D)

org_2D = torch.tensor([])
org_3D = torch.tensor([])
sha_2D = torch.tensor([])
sha_3D = torch.tensor([])

lengthhh = len(file_names_2D)

total_nans = 0

for idx in range(lengthhh):
    print(f"{idx}/{lengthhh}")
    data_2d = np.genfromtxt(os.path.join(directory_2D, file_names_2D[idx]), delimiter=",")
    data_3d = np.genfromtxt(os.path.join(directory_3D, file_names_3D[idx]), delimiter=",")

    #Add row of zeros
    data_2d = np.hstack((data_2d, np.zeros((data_2d.shape[0],1))))

    #Normalize 3D
    data_3d[:, 0] -= np.min(data_3d[:, 0])
    data_3d[:, 1] -= np.min(data_3d[:, 1])
    data_3d[:, 2] -= np.min(data_3d[:, 2])

    #Convert to spherical:
    origin_2D, shape_2D = convert_to_spherical(data_2d, tensor=True)
    origin_3D, shape_3D = convert_to_spherical(data_3d, tensor=True)

    origin_2D = torch.reshape(origin_2D, (1,3,1)).float()
    origin_3D = torch.reshape(origin_3D, (1,3,1)).float()

    shape_2D = torch.unsqueeze(torch.transpose(shape_2D, 0, 1), dim=0).float()
    shape_3D = torch.unsqueeze(torch.transpose(shape_3D, 0, 1), dim=0).float()

    if torch.sum(torch.isnan(shape_2D)) >= 1 or torch.sum(torch.isnan(shape_3D)) >= 1:
        print(f"Found nan in {idx}")
        total_nans += torch.sum(torch.isnan(shape_2D)) + torch.sum(torch.isnan(shape_3D))
    else:
        org_2D = torch.cat((org_2D, origin_2D), dim=0)
        org_3D = torch.cat((org_3D, origin_3D), dim=0)
        sha_2D = torch.cat((sha_2D, shape_2D), dim=0)
        sha_3D = torch.cat((sha_3D, shape_3D), dim=0)

torch.save(org_2D, 'D:\\CTA data\\origin_2D_interp_353.pt')
torch.save(org_3D, 'D:\\CTA data\\origin_3D_interp_353.pt')
torch.save(sha_2D, 'D:\\CTA data\\shape_2D_interp_353.pt')
torch.save(sha_3D, 'D:\\CTA data\\shape_3D_interp_353.pt')
print(f"Done, total nans: {total_nans}")
