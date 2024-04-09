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

warnings.filterwarnings("ignore")
#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

offset_list = np.genfromtxt("D:\\CTA data\\Offset_deformations.txt", delimiter=",")

class CenterlineDataset(Dataset):
    def __init__(self, data_dir_2D, data_dir_3D, transform=None):
        self.data_dir_2D = data_dir_2D
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.file_list_2D = os.listdir(data_dir_2D)
        self.file_list_3D = os.listdir(data_dir_3D)

    def __len__(self):
        return len(self.file_list_2D)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Load from file
        data_2d = np.genfromtxt(os.path.join(self.data_dir_2D, self.file_list_2D[idx]), delimiter=",")
        data_3d = np.genfromtxt(os.path.join(self.data_dir_3D, self.file_list_3D[idx]), delimiter=",")[1:, 1:4]

        # Add row of zeros
        data_2d = np.hstack((data_2d, np.zeros((data_2d.shape[0], 1))))

        # Normalize 3D
        data_3d[:, 0] -= np.min(data_3d[:, 0])
        data_3d[:, 1] -= np.min(data_3d[:, 1])
        data_3d[:, 2] -= np.min(data_3d[:, 2])

        # Convert to spherical:
        origin_2D, spherical_2D = convert_to_spherical(data_2d)
        origin_3D, spherical_3D = convert_to_spherical(data_3d)

        origin_2D = torch.reshape(origin_2D, (3, 1)).float()
        origin_3D = torch.reshape(origin_3D, (3, 1)).float()

        spherical_2D = torch.reshape(spherical_2D, (3, 349)).float()
        spherical_3D = torch.reshape(spherical_3D, (3, 349)).float()

        sample = {'origin_2D': origin_2D, 'origin_3D': origin_3D, 'shape_2D': spherical_2D, 'shape_3D': spherical_3D, 'offset': offset_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class CenterlineDatasetSpherical(Dataset):
    def __init__(self, base_dir, transform=None):
        self.origin_2D = torch.load(os.path.join(base_dir, "origin_2D.pt"))
        self.origin_3D = torch.load(os.path.join(base_dir, "origin_3D.pt"))
        self.shape_2D = torch.load(os.path.join(base_dir, "shape_2D.pt"))
        self.shape_3D = torch.load(os.path.join(base_dir, "shape_3D.pt"))
        self.transform = transform
    def __len__(self):
        return self.shape_2D.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'origin_2D': self.origin_2D[idx], 'origin_3D': self.origin_3D[idx], 'shape_2D': self.shape_2D[idx], 'shape_3D': self.shape_3D[idx], 'offset': offset_list[idx]}
        # sample = (self.origin_2D[idx], self.origin_3D[idx], self.shape_2D[idx], self.shape_3D[idx], offset_list[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

model = CARNet()

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

criterion = nn.MSELoss()  # put loss function we have here
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust hyperparameters according to paper


def train_model(model, criterion, optimizer, train_loader, num_epochs=186):
    for epoch in range(num_epochs):
        # model.train()
        running_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, input in loop:
            # Get data to cuda
            # origin_3D = origin_3D.to(device)
            # shape_3D = shape_3D.to(device)
            # origin_2D = origin_2D.to(device)
            # shape_2D = shape_2D.to(device)
            # offset = offset.to(device)

            #Set gradient to zero
            optimizer.zero_grad()

            #Forward pass
            outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
            loss = criterion(torch.Tensor(input['offset']), torch.Tensor(input['offset']))
            loss = Variable(loss, requires_grad=True)

            #Backward pass
            loss.backward()

            #Optimize
            optimizer.step()
            
            running_loss += loss.item() * input['origin_3D'].shape[0]

            #Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
        
    return model

# Load the data:
# train_dataset = CenterlineDataset(data_dir_2D="D:\\CTA data\\Segments_deformed_2\\", data_dir_3D="D:\\CTA data\\Segments renamed\\")
train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Train the model
trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=25)
