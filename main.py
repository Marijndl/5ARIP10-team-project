import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CARNet
import numpy as np

class VesselDataset(Dataset):
    def __init__(self, data_paths):
    self.data_paths = data_paths
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = 
        return data


model = CARNet()
criterion = nn.MSELoss()  # put loss function we have here
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust hyperparameters according to paper


def train_model(model, criterion, optimizer, train_loader, num_epochs=186):
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        
        for inputs in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
        
    return model



train_dataset = VesselDataset(data_paths=["your/data/path"], transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=25)
