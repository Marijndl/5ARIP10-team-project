import torch
from torch import nn, optim
from model import CARNet
from spherical_coordinates import *
from tqdm import tqdm
from data_loaders import *


class mPD_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def cartesian_tensor(self, origin, spherical):
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

        shape_cartesian = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), dim=1)
        shape_cartesian.retain_grad()
        full = torch.cat((origin.clone(), shape_cartesian), dim=2)
        full.retain_grad()

        cartesian = torch.matmul(full, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))).to(device))
        cartesian.retain_grad()
        return cartesian

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
        # Add deformation to 3D line
        spherical_3D_deformed = spherical_3D.clone()
        spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], deformation_field)

        # Convert back to cartesian domain
        deformed_cart = self.cartesian_tensor(origin_3D, spherical_3D_deformed)
        original_cart = self.cartesian_tensor(origin_2D, spherical_2D)

        # Calculate the loss
        loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))
        # loss = torch.sum(torch.mean(torch.sqrt(torch.sum(torch.square(deformed_cart - original_cart), dim=1)), dim=1), dim=0)
        loss.retain_grad()
        return loss

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize model weights:
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)

    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)

model = CARNet()
model = CARNet().to(device)
model = model.apply(weights_init)
criterion = mPD_loss_2()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjust hyperparameters according to paper

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

init_weigts = list(model.parameters())[-1].clone().detach()
print("Initial weights:", init_weigts)

def train_model(model, criterion, optimizer, train_loader, num_epochs=186):
    for epoch in range(num_epochs):
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
            outputs = model(input['origin_3D'].to(device), input['shape_3D'].to(device), input['origin_2D'].to(device), input['shape_2D'].to(device))
            outputs.requires_grad_(True)
            outputs.retain_grad()

            #Convert to cartesian and calculate loss:
            loss = criterion(input['origin_3D'].to(device), input['shape_3D'].to(device), input['origin_2D'].to(device), input['shape_2D'].to(device), outputs)

            #Backward pass
            loss.backward()

            #Optimize
            optimizer.step()

            running_loss += loss.clone().item() * input['origin_3D'].shape[0]

            #Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

    return model

if __name__ == "__main__":

    # Load the data:
    train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Train the model
    trained_model = train_model(model, criterion, optimizer, train_loader, num_epochs=20)

    # Save the weights
    torch.save(trained_model.state_dict(), "D:\\CTA data\\models\\CAR-Net-256-20.pth")

