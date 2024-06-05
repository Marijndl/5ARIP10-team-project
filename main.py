import pickle
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
from helper_functions import *
import torch
from tqdm import tqdm


class mPD_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def cartesian_tensor(self, origin, spherical):
        """
        Converts spherical coordinates to Cartesian coordinates and concatenates them with the origin tensor.

        Parameters:
        - origin: Original tensor
        - spherical: Spherical coordinates tensor

        Returns:
        - cartesian: Cartesian coordinates tensor
        """
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
        """
        Computes the smoothness loss based on the deformation field gradients.

        Parameters:
        - deformation_field: The deformation field tensor

        Returns:
        - smooth_loss: Smoothness loss value
        """
        def gradient(tensor):
            grad_y = tensor[:, :, 1:] - tensor[:, :, :-1]
            grad_x = tensor[:, 1:, :] - tensor[:, :-1, :]
            return grad_y, grad_x

        grad_y, grad_x = gradient(deformation_field)

        smooth_loss = torch.sum(grad_y**2) + torch.sum(grad_x**2)
        return smooth_loss

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field, alpha=0.02):
        """
        Forward pass for the custom loss function.

        Parameters:
        - origin_3D: Original 3D tensor
        - spherical_3D: Spherical 3D coordinates
        - origin_2D: Original 2D tensor
        - spherical_2D: Spherical 2D coordinates
        - deformation_field: Deformation field tensor
        - alpha: Smoothing factor

        Returns:
        - total_loss: Total loss value combining difference and smoothness loss
        """
        spherical_3D_deformed = spherical_3D.clone()
        spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], deformation_field)

        deformed_cart = self.cartesian_tensor(origin_3D, spherical_3D_deformed)
        original_cart = self.cartesian_tensor(origin_2D, spherical_2D)

        diff = deformed_cart - original_cart
        diff_xy = diff[:, :2, :].clone().requires_grad_(True)

        loss = torch.sum(torch.mean(torch.sum(torch.abs(diff_xy), dim=1), dim=1))
        smooth_loss = self.smoothness_loss(deformation_field)

        total_loss = loss + alpha * smooth_loss
        return total_loss

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 5e9 else "cpu")

def weights_init(m):
    """
    Initializes the weights of the model.

    Parameters:
    - m: Model layer
    """
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data, mean=0.0, std=1)

def train_model(model, criterion, optimizer, train_loader, val_loader,batch_size, learning_rate ,num_epochs=186,
                model_save_name="CAR-Net-val.pth", checkpoint_path=None, new_learning_rate_factor=None, scheduler=None, smoothing=0.02, scheduler_type='None'):
    """
    Function to train the model, save the best model, and return the training and validation losses. When a
    checkpoint_path is provided the training will resume from the last checkpoint. The model will be saved when the
    training is stopped, and the best model will be saved when the validation loss is lower than the previous best
    validation loss.


    Parameters:
    - model: the model to train
    - criterion: the loss function
    - optimizer: the optimizer
    - train_loader: DataLoader for the training set
    - val_loader: DataLoader for the validation set
    - num_epochs: the number of epochs to train
    - model_save_name: the name of the file to save the model
    - checkpoint_path: the path to the checkpoint file
    - new_learning_rate_factor: the factor to adjust the learning rate by
    """

    model_save_name_val = model_save_name.split(".")[0] + "_val"
    print(f"The best validation model will be saved as {model_save_name_val}, and the final model as {model_save_name}")

    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epoch_train_loss = float('inf')
    epoch_val_loss = float('inf')

    global stop_training

    if checkpoint_path and os.path.exists(checkpoint_path + ".pth"):
        model, optimizer, start_epoch, train_loss, val_loss, scheduler = load_model(model, optimizer, checkpoint_path, scheduler)
        print(f"Checkpoint loaded: resuming from epoch {start_epoch} with last train loss:{train_loss}, last val loss: {val_loss}")
    else:
        print("No model checkpoint found, starting training from scratch")

    if new_learning_rate_factor:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate_factor * param_group['lr']
            print(f"Learning rate adjusted to {param_group['lr']}")

    #Load data on the GPU
    preloaded_train_batches = []
    for input in train_loader:
        input = {k: v.to(device) for k, v in input.items()}
        preloaded_train_batches.append(input)

    preloaded_val_batches = []
    for input in val_loader:
        input = {k: v.to(device) for k, v in input.items()}
        preloaded_val_batches.append(input)

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        total_samples = 0
        model.train()

        if stop_training:
            model_save_name_checkpoint = model_save_name.split(".")[0] + "_checkpoint"
            tqdm.write(f"Stopped training at epoch {epoch + 1}, saving the model as a checkpoint with the name: {model_save_name_checkpoint}")
            save_model(model, epoch, batch_size, learning_rate, smoothing, optimizer,epoch_train_loss,  epoch_val_loss, best_val_loss, model_save_name_checkpoint, scheduler=scheduler, scheduler_type=scheduler_type)
            break

        loop = tqdm(enumerate(preloaded_train_batches), total=len(preloaded_train_batches), leave=True)
        for batch_idx, input in loop:
            # Set gradient to zero
            optimizer.zero_grad()

            # Forward pass
            input['origin_3D'].requires_grad_(True)
            input['shape_3D'].requires_grad_(True)
            input['origin_2D'].requires_grad_(True)
            input['shape_2D'].requires_grad_(True)
            batch_size = input['origin_3D'].size(0)  # Assuming batch size is the same across all inputs

            outputs = model(input['origin_3D'],
                            input['shape_3D'],
                            input['origin_2D'],
                            input['shape_2D'])
            outputs.requires_grad_(True)

            # Calculate loss
            loss = criterion(input['origin_3D'],
                             input['shape_3D'],
                             input['origin_2D'],
                             input['shape_2D'],
                             outputs, alpha=smoothing)

            loss.backward()

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            total_samples += batch_size  # Accumulate the number of samples

            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], LR: {optimizer.param_groups[0]['lr']:.6f}")
            loop.set_postfix(loss=batch_loss / batch_size)



        epoch_train_loss = epoch_loss / total_samples
        train_losses.append(epoch_train_loss)

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for input in preloaded_val_batches:
                outputs = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])
                val_loss = criterion(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'],
                                     outputs)
                val_total_loss += val_loss.item()

        #Update current loss statistics
        epoch_val_loss = val_total_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        tqdm.write(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

        #Update learning rate based on scheduler
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(epoch_val_loss)
        elif scheduler_type == 'StepLR':
            scheduler.step()

        #Save model if it has improved
        if epoch_val_loss < best_val_loss:
            print(f"New best validation loss, saving the model as {model_save_name_val}")
            best_val_loss = epoch_val_loss
            if overwrite_model(model_save_name_val, best_val_loss):
                save_model(model, epoch+1, batch_size, learning_rate, smoothing,  optimizer, epoch_train_loss,  epoch_val_loss, best_val_loss, model_save_name_val, scheduler=scheduler, scheduler_type=scheduler_type)


    if stop_training == False:
        print(f"Training completed, saving the model as {model_save_name}")
        # if the model already exists, read the best_val_loss from the file, if it is lower than the current
        # best_val_loss, don't overwrite it
        if overwrite_model(model_save_name, best_val_loss):
            save_model(model, epoch+1, batch_size, learning_rate, smoothing,  optimizer, epoch_train_loss,  epoch_val_loss, best_val_loss, model_save_name, scheduler=scheduler, scheduler_type=scheduler_type)

    return train_losses, val_losses


stop_training = False
def on_key_press(event):
    global stop_training
    if event.name == 'K':
        print("ey press 'K' detected, stopping training at the end of the epoch")
        stop_training = True

# Register the key press event
keyboard.on_press(on_key_press)

# Hyperparameters
batch_size = 256
learning_rate = 0.02
optimizer_name = 'Adam'
number_of_epochs = 150
smoothing = 0.02
use_scheduler = True

# Learning rate scheduler
schedule_type = 'StepLR' # Use 'ReduceLROnPlateau' or 'StepLR'
scheduler_step_size = 6
scheduler_gamma = 0.1

load_best_params = True    # Load the best parameters found by the Bayesian optimization from the best_params.txt file
model_save_name = "CAR-Net-Optimizer_trial0"     # Name of the model to save
checkpoint_path = f"D:\\CTA data\\models\\{model_save_name}_checkpoint"

if __name__ == "__main__":

    #Initialize model
    model = CARNet().to(device)
    model.apply(weights_init)
    criterion = mPD_loss_2()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    if load_best_params:
        with open("Models\\Hyperparameter_optimization\\best_params.txt", "r") as f:
            best_learning_rate = float(f.readline().split(": ")[1])
            best_optimizer_name = f.readline().split(": ")[1].strip()
            best_batch_size = int(f.readline().split(": ")[1])
            best_smoothing = float(f.readline().split(": ")[1])
            best_scheduler = f.readline().split(": ")[1].strip()
            batch_size = best_batch_size
            learning_rate = best_learning_rate
            optimizer_name = best_optimizer_name
            smoothing = best_smoothing
            schedule_type = best_scheduler

    #Apply a scheduler
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if use_scheduler:
        if schedule_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        elif schedule_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        elif schedule_type == 'None':
            scheduler = None
        else:
            assert False, "Invalid scheduler type"
    else:
        scheduler = None
        schedule_type = 'None'

    #Initialize data
    print(f"Training with learning rate: {learning_rate}, optimizer: {optimizer_name}, batch size: {batch_size}",
          f"smoothing: {smoothing}, scheduler: {schedule_type}")
    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_loader, val_loader, test_loader = create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                            batch_size=batch_size, shuffle_train=True)

    #Train the model
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader,batch_size, learning_rate,
                                                          num_epochs=number_of_epochs,
                                                          model_save_name=model_save_name,
                                                          checkpoint_path=checkpoint_path,
                                                          new_learning_rate_factor=None,
                                                          scheduler=scheduler,
                                                          smoothing=smoothing,
                                                          scheduler_type=schedule_type)

    #Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

