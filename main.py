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

def convert_to_projection(origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
    deformed = torch.tensor([])
    original = torch.tensor([])

    # Add deformation to 3D line
    spherical_3D[:, 1:, :] += deformation_field

    for idx in range(deformation_field.shape[0]):
        # Convert back to cartesian
        cartesian_2D = convert_back(torch.transpose(origin_2D[idx], 0, 1).detach().numpy(),
                                    torch.transpose(spherical_2D[idx], 0, 1).detach().numpy()).squeeze()
        cartesian_3D = convert_back(torch.transpose(origin_3D[idx], 0, 1).detach().numpy(),
                                    torch.transpose(spherical_3D[idx], 0, 1).detach().numpy()).squeeze()

        # Project to 2D
        cartesian_3D[:, 2] = np.zeros(cartesian_3D.shape[0])

        original = torch.cat((original, torch.unsqueeze(torch.transpose(torch.from_numpy(cartesian_2D).float(), 0, 1), dim=0)), dim=0)
        deformed = torch.cat((deformed, torch.unsqueeze(torch.transpose(torch.from_numpy(cartesian_3D).float(), 0, 1), dim=0)), dim=0)

    return deformed, original

def mPD_loss(deformed, original):
    loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed - original), dim=1), dim=1))
    return loss

torch.cuda.empty_cache()

warnings.filterwarnings("ignore")
offset_list = np.genfromtxt("D:\\CTA data\\Offset_deformations.txt", delimiter=",")

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize model:
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)
        # xavier(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)
        # xavier(m.bias.data)


model = CARNet()
# model = CARNet().to(device)
# model = model.apply(weights_init)
criterion = nn.MSELoss()  # put loss function we have here
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust hyperparameters according to paper

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

def train_model(model, criterion, optimizer, train_loader, num_epochs=186):
    for epoch in range(num_epochs):
        # model.train()
        running_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, input in loop:
            #Set gradient to zero
            optimizer.zero_grad()

            #Forward pass
            outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
            # outputs = model(input['origin_3D'].to(device), input['shape_3D'].to(device), input['origin_2D'].to(device), input['shape_2D'].to(device))
            deformed, original = convert_to_projection(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'], outputs)

            loss = criterion(deformed, original)
            # loss = mPD_loss(deformed, original)
            loss = Variable(loss, requires_grad=True)
            # print(loss.item())

            #Backward pass
            loss.backward()

            #Optimize
            optimizer.step()
            
            running_loss += loss.item() * input['origin_3D'].shape[0]

            #Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}\n')
        
    return model

# Load the data:
# train_dataset = CenterlineDataset(data_dir_2D="D:\\CTA data\\Segments_deformed_2\\", data_dir_3D="D:\\CTA data\\Segments renamed\\")
train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Train the model
trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=25)
