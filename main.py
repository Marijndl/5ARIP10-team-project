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

warnings.filterwarnings("ignore")
offset_list = np.genfromtxt("D:\\CTA data\\Offset_deformations.txt", delimiter=",")

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize model:
model = CARNet().to(device)
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
            outputs = model(input['origin_3D'].to(device), input['shape_3D'].to(device), input['origin_2D'].to(device), input['shape_2D'].to(device))
            loss = criterion(torch.Tensor(input['offset'].to(device)), torch.Tensor(input['offset'].to(device)))
            loss = Variable(loss, requires_grad=True)

            #Backward pass
            loss.backward()

            #Optimize
            optimizer.step()
            
            running_loss += loss.item() * input['origin_3D'].shape[0]

            #Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs-1}]")
            loop.set_postfix(loss = loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch {epoch+1}/{num_epochs-1}, Loss: {epoch_loss:.4f}\n')
        
    return model

# Load the data:
# train_dataset = CenterlineDataset(data_dir_2D="D:\\CTA data\\Segments_deformed_2\\", data_dir_3D="D:\\CTA data\\Segments renamed\\")
train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Train the model
trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=25)
