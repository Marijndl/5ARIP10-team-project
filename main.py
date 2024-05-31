import keyboard
import optuna
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

        diff = deformed_cart - original_cart
        diff_xy = diff[:, :2, :].clone().requires_grad_(True)

        loss = torch.sum(torch.mean(torch.sum(torch.abs(diff_xy), dim=1), dim=1))
        smooth_loss = self.smoothness_loss(deformation_field)

        total_loss = loss + 0.02 * smooth_loss
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


total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=186, model_save_name="CAR-Net-val.pth"):
    model_save_name_val = model_save_name.split(".")[0] + "_val"
    print(f"Saving the best validation model as {model_save_name_val}")
    train_losses = []
    val_losses = []
    global stop_training
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_samples = 1
        batch_losses = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        model.train()
        if stop_training:
            tqdm.write(f"Stopped training at epoch {epoch + 1}")
            break

        for batch_idx, input in loop:
            # Set gradient to zero
            optimizer.zero_grad()

            # Forward pass
            input['origin_3D'].requires_grad_(True)
            input['shape_3D'].requires_grad_(True)
            input['origin_2D'].requires_grad_(True)
            input['shape_2D'].requires_grad_(True)
            batch_size = input['origin_3D'].size(0)  # Assuming batch size is the same across all inputs

            outputs = model(input['origin_3D'].to(device),
                            input['shape_3D'].to(device),
                            input['origin_2D'].to(device),
                            input['shape_2D'].to(device))
            outputs.requires_grad_(True)

            # Calculate loss
            loss = criterion(input['origin_3D'].to(device),
                             input['shape_3D'].to(device),
                             input['origin_2D'].to(device),
                             input['shape_2D'].to(device),
                             outputs)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Calculate batch loss
            batch_loss = loss.item()
            epoch_loss += batch_loss
            total_samples += batch_size  # Accumulate the number of samples

            # Update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], LR: {optimizer.param_groups[0]['lr']:.6f}")
            loop.set_postfix(loss=batch_loss / batch_size)
            # print(list(model.parameters())[-1].clone().detach())

        # Update learning rate scheduler
        gradual.step()

        # Calculate average loss per sample
        epoch_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)


        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for input in val_loader:
                input['origin_3D'] = input['origin_3D'].to(device)
                input['shape_3D'] = input['shape_3D'].to(device)
                input['origin_2D'] = input['origin_2D'].to(device)
                input['shape_2D'] = input['shape_2D'].to(device)
                outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
                val_loss = criterion(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'], outputs)
                val_total_loss += val_loss.item()

        epoch_val_loss = val_total_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        tqdm.write(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
        if val_losses[-1] == min(val_losses): # Save the model with the lowest validation loss
            torch.save(model.state_dict(), f"D:\\CTA data\\models\\{model_save_name_val}.pth")
            print("Model saved")

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
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')
    return test_loss

stop_training = False
def on_key_press(event):
    global stop_training
    if event.name == 'q':
        print("Key press detected, stopping training at the end of the epoch")
        stop_training = True

# Register the key press event
keyboard.on_press(on_key_press)


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # Create model, criterion, and optimizer
    model = CARNet().to(device)
    model.apply(weights_init)
    criterion = mPD_loss_2()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Load dataset and create DataLoader
    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_loader, val_loader, test_loader = create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                            batch_size=batch_size, shuffle_train=True)

    # Train model
    trained_model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader,
                                                          num_epochs=30)

    # Return the last validation loss as the objective to minimize
    return min(val_losses)

BayesianOptimization = False
model_save_name = "CAR-Net-optimizedv3.pth"

def optimization():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Use the best parameters to retrain the model
    best_learning_rate = trial.params['learning_rate']
    best_optimizer_name = trial.params['optimizer']
    best_batch_size = trial.params['batch_size']
    # save best parameters
    with open("D:\\CTA data\\models\\best_params.txt", "w") as f:
        f.write(f"Best learning rate: {best_learning_rate}\n")
        f.write(f"Best optimizer: {best_optimizer_name}\n")
        f.write(f"Best batch size: {best_batch_size}\n")


if __name__ == "__main__":
    if BayesianOptimization:
        optimization()
    else:
        with open("D:\\CTA data\\models\\best_params.txt", "r") as f:
            best_learning_rate = float(f.readline().split(": ")[1])
            best_optimizer_name = f.readline().split(": ")[1].strip()
            best_batch_size = int(f.readline().split(": ")[1])

    model = CARNet().to(device)
    model.apply(weights_init)
    criterion = mPD_loss_2()
    optimizer = getattr(optim, best_optimizer_name)(model.parameters(), lr=best_learning_rate)
    gradual = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_loader, val_loader, test_loader = create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                            batch_size=best_batch_size, shuffle_train=True)

    trained_model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader,
                                                          num_epochs=150, model_save_name=model_save_name)

    torch.save(trained_model.state_dict(), f"D:\\CTA data\\models\\{model_save_name}")
    print("Model saved")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    test_loss = test_model(trained_model, criterion, test_loader)


