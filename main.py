import torch
from torch import nn, optim
from model import CARNet
from spherical_coordinates import *
from tqdm import tqdm
from data_loaders import *
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class mPD_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def cartesian_tensor(self, origin, spherical):
        r = spherical[:, 0, :].clone().requires_grad_(True)
        theta = spherical[:, 1, :].clone().requires_grad_(True)
        phi = spherical[:, 2, :].clone().requires_grad_(True)

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        shape_cartesian = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), dim=1).requires_grad_(True)
        full = torch.cat((origin.clone(), shape_cartesian), dim=2).requires_grad_(True)

        cartesian = torch.matmul(full, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))).to(device)).requires_grad_(True)
        return cartesian

    def smoothness_loss(self, deformation_field):
        def gradient(tensor):
            grad_y = tensor[:, :, 1:] - tensor[:, :, :-1]
            grad_x = tensor[:, 1:, :] - tensor[:, :-1, :]
            return grad_y, grad_x
        
        grad_y, grad_x = gradient(deformation_field)
        
        smooth_loss = torch.sum(grad_y**2) + torch.sum(grad_x**2)
        return smooth_loss

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
        spherical_3D_deformed = spherical_3D.clone()
        spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], deformation_field)

        deformed_cart = self.cartesian_tensor(origin_3D, spherical_3D_deformed)
        original_cart = self.cartesian_tensor(origin_2D, spherical_2D)

        loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))
        smooth_loss = self.smoothness_loss(deformation_field)
        
        total_loss = loss + smooth_loss
        return total_loss

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)

model = CARNet().to(device)
model.apply(weights_init)
criterion = mPD_loss_2()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=186):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, input in loop:
            optimizer.zero_grad()

            input['origin_3D'] = input['origin_3D'].to(device).requires_grad_(True)
            input['shape_3D'] = input['shape_3D'].to(device).requires_grad_(True)
            input['origin_2D'] = input['origin_2D'].to(device).requires_grad_(True)
            input['shape_2D'] = input['shape_2D'].to(device).requires_grad_(True)

            outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])

            loss = criterion(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'], outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input['origin_3D'].shape[0]
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input in val_loader:
                input['origin_3D'] = input['origin_3D'].to(device)
                input['shape_3D'] = input['shape_3D'].to(device)
                input['origin_2D'] = input['origin_2D'].to(device)
                input['shape_2D'] = input['shape_2D'].to(device)
                outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
                loss = criterion(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'], outputs)
                val_loss += loss.item() * input['origin_3D'].shape[0]

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}')

    return model, train_losses, val_losses

def test_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input in test_loader:
            input['origin_3D'] = input['origin_3D'].to(device)
            input['shape_3D'] = input['shape_3D'].to(device)
            input['origin_2D'] = input['origin_2D'].to(device)
            input['shape_2D'] = input['shape_2D'].to(device)
            outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
            loss = criterion(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'], outputs)
            test_loss += loss.item() * input['origin_3D'].shape[0]
    
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')
    return test_loss

if __name__ == "__main__":
    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    trained_model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)

    torch.save(trained_model.state_dict(), "D:\\CTA data\\models\\CAR-Net-256-23.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    test_loss = test_model(trained_model, criterion, test_loader)
