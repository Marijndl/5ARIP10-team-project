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

    # spherical_3D = spherical_3D.clone()

    # Add deformation to 3D line
    spherical_3D[:, 1:, :] += deformation_field

    for idx in range(deformation_field.shape[0]):
        # Convert back to cartesian
        cartesian_2D = convert_back_tensors(origin_2D[idx].detach(), spherical_2D[idx].detach())
        cartesian_3D = convert_back_tensors(origin_3D[idx].detach(), spherical_3D[idx].detach())

        # Project to 2D
        cartesian_3D[2, :] = torch.zeros(cartesian_3D.shape[1])

        original = torch.cat((original, cartesian_2D.unsqueeze(dim=0)), dim=0)
        deformed = torch.cat((deformed, cartesian_3D.unsqueeze(dim=0)), dim=0)

    return deformed, original

class mPD_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
        deformed = torch.tensor([], requires_grad=True)
        original = torch.tensor([], requires_grad=True)
        # Add deformation to 3D line
        spherical_3D[:, 1:, :] += deformation_field

        for idx in range(deformation_field.shape[0]):
            # Convert back to cartesian
            cartesian_2D = convert_back_tensors(origin_2D[idx], spherical_2D[idx])
            cartesian_3D = convert_back_tensors(origin_3D[idx], spherical_3D[idx])

            # Project to 2D
            cartesian_3D[2, :] = torch.zeros(cartesian_3D.shape[1], requires_grad=True)

            original = torch.cat((original, cartesian_2D.unsqueeze(dim=0)), dim=0)
            deformed = torch.cat((deformed, cartesian_3D.unsqueeze(dim=0)), dim=0)

        loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed - original), dim=1), dim=1))
        return loss

def cal_cart_tensor(origin, spherical):
    r = spherical[:, 0, :].clone()
    theta = spherical[:, 1, :].clone()
    phi = spherical[:, 2, :].clone()

    r.retain_grad()
    theta.retain_grad()
    phi.retain_grad()

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    x.retain_grad()
    y.retain_grad()
    z.retain_grad()

    shape = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), dim=1)
    shape.retain_grad()
    full = torch.cat((origin.clone(), shape), dim=2)
    full.retain_grad()

    cartesian = torch.matmul(full, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))))
    cartesian.retain_grad()
    return cartesian

class mPD_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
        # Add deformation to 3D line
        spherical_3D_deformed = spherical_3D.clone()
        spherical_3D_deformed[:, 1:, :] = spherical_3D_deformed[:, 1:, :] + deformation_field

        original_cart = cal_cart_tensor(origin_2D, spherical_2D)
        deformed_cart = cal_cart_tensor(origin_3D, spherical_3D_deformed)

        loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))
        loss.retain_grad()
        return loss

torch.autograd.set_detect_anomaly(True)
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
model = model.apply(weights_init)
# criterion = nn.MSELoss()  # put loss function we have here
criterion = mPD_loss_2()
optimizer = optim.Adam(model.parameters(), lr=1, weight_decay=1e-4)  # Adjust hyperparameters according to paper

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

init_weigts = list(model.parameters())[-1].clone().detach()
print("Initial weights:", init_weigts)

def train_model(model, criterion, optimizer, train_loader, num_epochs=186):
    for epoch in range(num_epochs):
        # model.train()
        running_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, input in loop:
            #Set gradient to zero
            optimizer.zero_grad()

            #Forward pass
            input['origin_3D'].requires_grad_(True)
            input['shape_3D'].requires_grad_(True)
            input['origin_2D'].requires_grad_(True)
            input['shape_2D'].requires_grad_(True)
            input['origin_3D'].retain_grad()
            input['shape_3D'].retain_grad()
            input['origin_2D'].retain_grad()
            input['shape_2D'].retain_grad()
            outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
            outputs.requires_grad_(True)
            outputs.retain_grad()
            # outputs = model(input['origin_3D'].to(device), input['shape_3D'].to(device), input['origin_2D'].to(device), input['shape_2D'].to(device))
            # deformed, original = convert_to_projection_old(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'], outputs.detach())

            spherical_3D_deformed = input['shape_3D'].clone()
            spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], outputs)

            r = spherical_3D_deformed[:, 0, :].clone()
            theta = spherical_3D_deformed[:, 1, :].clone()
            phi = spherical_3D_deformed[:, 2, :].clone()

            r.retain_grad()
            theta.retain_grad()
            phi.retain_grad()

            x = r * torch.sin(theta) * torch.cos(phi)
            y = r * torch.sin(theta) * torch.sin(phi)
            z = r * torch.cos(theta)

            x.retain_grad()
            y.retain_grad()
            z.retain_grad()

            shape = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), dim=1)
            shape.retain_grad()
            full = torch.cat((input['origin_3D'].clone(), shape), dim=2)
            full.retain_grad()

            original_cart = torch.matmul(full, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))))
            original_cart.retain_grad()

            # ___--------------------------------------------------

            r2 = input['shape_2D'][:, 0, :].clone()
            theta2 = input['shape_2D'][:, 1, :].clone()
            phi2 = input['shape_2D'][:, 2, :].clone()

            r2.retain_grad()
            theta2.retain_grad()
            phi2.retain_grad()

            x2 = r2 * torch.sin(theta2) * torch.cos(phi2)
            y2 = r2 * torch.sin(theta2) * torch.sin(phi2)
            z2 = r2 * torch.cos(theta2)

            x2.retain_grad()
            y2.retain_grad()
            z2.retain_grad()

            shape2 = torch.cat((x2.unsqueeze(dim=1), y2.unsqueeze(dim=1), z2.unsqueeze(dim=1)), dim=1)
            shape2.retain_grad()
            full2 = torch.cat((input['origin_2D'].clone(), shape2), dim=2)
            full2.retain_grad()

            deformed_cart = torch.matmul(full2, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))))
            deformed_cart.retain_grad()

            loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))

            #Backward pass
            loss.backward()
            # print(outputs.grad.shape)

            #Optimize
            optimizer.step()
            # print(list(model.parameters())[-1].grad)
            
            # running_loss += loss.detach() * input['origin_3D'].shape[0]

            #Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            print("Updated weights:", list(model.parameters())[-1].clone().detach())

    return model

# Load the data:
# train_dataset = CenterlineDataset(data_dir_2D="D:\\CTA data\\Segments_deformed_2\\", data_dir_3D="D:\\CTA data\\Segments renamed\\")
train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Train the model
trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=5)

# Save the weights
torch.save(trained_model.state_dict(), "D:\\CTA data\\models\\CAR-Net-256-defor.pth")

