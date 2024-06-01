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

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field, alpha=0.02):
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

import torch
from tqdm import tqdm


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=186,
                model_save_name="CAR-Net-val.pth", checkpoint_path=None, new_learning_rate_factor=None, scheduler=None, smoothing=0.02, scheduler_type='None'):
    """ Function to train the model, save the best model, and return the training and validation losses. When a
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
            save_model(model, epoch, optimizer,epoch_train_loss,  epoch_val_loss, best_val_loss, model_save_name_checkpoint, scheduler=scheduler, scheduler_type=scheduler_type)
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

        epoch_val_loss = val_total_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        tqdm.write(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(epoch_val_loss)
        elif scheduler_type == 'StepLR':
            scheduler.step()


        if epoch_val_loss < best_val_loss:
            print(f"New best validation loss, saving the model as {model_save_name_val}")
            best_val_loss = epoch_val_loss
            if overwrite_model(model_save_name_val, best_val_loss):
                save_model(model, epoch+1, optimizer, epoch_train_loss,  epoch_val_loss, best_val_loss, model_save_name_val, scheduler=scheduler, scheduler_type=scheduler_type)


    if stop_training == False:
        print(f"Training completed, saving the model as {model_save_name}")
        # if the model already exists, read the best_val_loss from the file, if it is lower than the current
        # best_val_loss, don't overwrite it
        if overwrite_model(model_save_name, best_val_loss):
            save_model(model, epoch, optimizer, epoch_train_loss,  epoch_val_loss, best_val_loss, model_save_name, scheduler=scheduler, scheduler_type=scheduler_type)

    return train_losses, val_losses

def overwrite_model(model_save_name, best_val_loss):
    overwrite = True
    if os.path.exists(f"D:\\CTA data\\models\\{model_save_name}.pth"):
        print(f"Model with the name {model_save_name} already exists, checking if the best validation loss is lower "
              f"than the new best validation loss")
        with open(f"D:\\CTA data\\models\\{model_save_name}_params.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Best validation loss" in line:
                    old_best_val_loss = float(line.split(": ")[1])
                    if old_best_val_loss < best_val_loss:
                        overwrite = False
                        print(f"Not overwriting the model, old best validation loss is lower: {old_best_val_loss}, "
                              f"current is: {best_val_loss}")
                    else:
                        overwrite = True
                        print(f"Overwriting the model, old best validation loss is higher: {old_best_val_loss}, current "
                              f"is: {best_val_loss}")
    else:
        overwrite = True
    return overwrite


def save_model(model, epoch, optimizer, train_loss, val_loss, best_val_loss, model_save_name, scheduler=None, scheduler_type='None'):
    if scheduler:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'scheduler_state_dict': scheduler.state_dict(),
        }, f"D:\\CTA data\\models\\{model_save_name}.pth")
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f"D:\\CTA data\\models\\{model_save_name}.pth")
    # create file with parameters and statistics
    with open(f"D:\\CTA data\\models\\{model_save_name}_params.txt", "w") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Train Loss: {train_loss}\n")
        f.write(f"Validation Loss: {val_loss}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Optimizer: {optimizer}\n")
        if scheduler_type == 'ReduceLROnPlateau':
            f.write(f"Scheduler {scheduler_type}: mode: {scheduler.mode}, factor: {scheduler.factor}, patience: {scheduler.patience}\n, threshold: {scheduler.threshold}\n")
        elif scheduler_type == 'StepLR':
            f.write(f"Scheduler {scheduler_type}: step size: {scheduler.step_size}, gamma: {scheduler.gamma}\n")
        else:
            f.write("No scheduler used\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Initial learning rate: {learning_rate}\n")
        f.write(f"Smoothing: {smoothing}\n")
        f.write(f"Best validation loss: {best_val_loss}\n")


def load_model(model, optimizer, path_checkpoint, scheduler):
    checkpoint = torch.load(f"{path_checkpoint}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        scheduler = None

    return model, optimizer, epoch, train_loss, val_loss, scheduler


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
    if event.name == 'K':
        print("ey press 'K' detected, stopping training at the end of the epoch")
        stop_training = True

# Register the key press event
keyboard.on_press(on_key_press)

def evaluate_model(model, test_loader, loss):
    model.eval()
    all_distances = []

    for batch in test_loader:
        batch['origin_3D'] = batch['origin_3D'].to(device)
        batch['shape_3D'] = batch['shape_3D'].to(device)
        batch['origin_2D'] = batch['origin_2D'].to(device)
        batch['shape_2D'] = batch['shape_2D'].to(device)


        deformation_field = model(batch['origin_3D'], batch['shape_3D'], batch['origin_2D'], batch['shape_2D'])


        spherical_3D_deformed = batch['shape_3D'].clone()
        spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], deformation_field)

        # Convert back to cartesian domain
        deformed_cart_3D = loss.cartesian_tensor(batch['origin_3D'], spherical_3D_deformed)
        original_cart_3D = loss.cartesian_tensor(batch['origin_3D'], batch['shape_3D'])
        original_cart_2D = loss.cartesian_tensor(batch['origin_2D'], batch['shape_2D'])

        deformed_3D = deformed_cart_3D.clone().detach().cpu().numpy()
        original_3D = original_cart_3D.clone().detach().cpu().numpy()
        original_2D = original_cart_2D.clone().detach().cpu().numpy()
        # Remove z component
        difference = abs(deformed_3D - original_2D)
        difference = difference[:, :2, :]

        # pythagorean theorem
        distances = np.mean(np.sqrt(np.sum(difference ** 2, axis=1)), axis=1)
        all_distances.extend(distances)
    plot_3D_centerline(original_3D, deformed_3D, original_2D, distances, -1)
    print(f"Number of samples: {len(all_distances)}")

    mPD = np.mean(all_distances)
    std_mPD = np.std(all_distances)
    # print(f"Mean time per sample: {np.mean(times):.2f}")
    print(f"Mean Projection Distance: {mPD:.2f}")
    print(f"Standard deviation: {std_mPD:.2f}")
    return mPD, std_mPD

def plot_3D_centerline(original_3D, deformed_3D, original_2D, distances, idx=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(f"Mean Projection Distance: {distances[idx]}")

    # Extract x, y, z coordinates from the input array
    org_x = original_2D[idx, 0, :]
    org_y = original_2D[idx, 1, :]

    def_x = deformed_3D[idx, 0, :]
    def_y = deformed_3D[idx, 1, :]
    def_z = deformed_3D[idx, 2, :]
    # def_z = deformed_3D[idx, 2, :]
    # def_z = np.zeros(original_2D[idx, 2, :].shape)

    org_x_3 = original_3D[idx, 0, :]
    org_y_3 = original_3D[idx, 1, :]
    org_z_3 = original_3D[idx, 2, :]

    # Plot the lines connecting the points
    ax.plot(org_x, org_y)
    ax.plot(def_x, def_y)
    ax.plot(org_x_3, org_y_3, org_z_3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # set the aspect ratio of the plot to be equal
    ax.set_aspect('equal', 'box')

    ax.legend(
        ['Original 2D centerline segment', 'Deformed 3D centerline segment', 'Original 3D centerline segment'])
    plt.show()


def objective(trial):
    # Suggest hyperparameters
    global batch_size, learning_rate, optimizer_name, optimization_epochs, smoothing, schedule_type, stop_training
    learning_rate = trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    smoothing = trial.suggest_float('smoothing', 0.001, 0.1, log=True)
    schedule_type = trial.suggest_categorical('schedule_type', ['ReduceLROnPlateau', 'StepLR', 'None'])


    # Create model, criterion, and optimizer
    model = CARNet().to(device)
    model.apply(weights_init)
    criterion = mPD_loss_2()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    if schedule_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=optimization_factor, patience=optimization_patience, verbose=True)
    elif schedule_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optimization_step_size, gamma=optimization_gamma)
    else:
        scheduler = None
    name = f"CAR-Net-Optimizer_trial{trial.number}"

    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_loader, val_loader, test_loader = create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                            batch_size=batch_size, shuffle_train=True)
    print(f"Training with learning rate: {learning_rate}, optimizer: {optimizer_name}, batch size: {batch_size}, smoothing: {smoothing}, scheduler: {schedule_type}")
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader,
                                                          num_epochs=optimization_epochs,
                                                          smoothing=smoothing,
                                                          scheduler=scheduler,
                                                          scheduler_type=schedule_type,
                                                          model_save_name=name,)
    validation_name = name + "_val"
    print(len(val_loader))
    best_model = load_model(model, optimizer, f"D:\\CTA data\\models\\{validation_name}", scheduler)[0]
    best_model.eval()
    mPd, std_mPd = evaluate_model(best_model, val_loader, criterion)
    stop_training = False
    return mPd



def optimization(trails=30):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trails)


    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    # save study
    with open("D:\\CTA data\\models\\study.pkl", "wb") as f:
        pickle.dump(study, f)

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Use the best parameters to retrain the model
    best_learning_rate = trial.params['learning_rate']
    best_optimizer_name = trial.params['optimizer']
    best_batch_size = trial.params['batch_size']
    best_smoothing = trial.params['smoothing']
    best_scheduler = trial.params['schedule_type']
    # save best parameters
    with open("D:\\CTA data\\models\\best_params.txt", "w") as f:
        f.write(f"Best learning rate: {best_learning_rate}\n")
        f.write(f"Best optimizer: {best_optimizer_name}\n")
        f.write(f"Best batch size: {best_batch_size}\n")
        f.write(f"Best smoothing: {best_smoothing}\n")
        f.write(f"Best scheduler: {trial.params['schedule_type']}\n")
    return best_learning_rate, best_optimizer_name, best_batch_size, best_smoothing, best_scheduler

# Hyperparameters
batch_size = 256
learning_rate = 0.02
optimizer_name = 'Adam'
number_of_epochs = 150
smoothing = 0.02
use_scheduler = True
schedule_type = 'StepLR' # Use 'ReduceLROnPlateau' or 'StepLR'


# Learning rate scheduler

scheduler_step_size = 10
scheduler_gamma = 0.1

BayesianOptimization = True  # Set to True to perform Bayesian optimization
number_of_trials = 1
optimization_epochs = 1
optimization_step_size = 6
optimization_gamma = 0.1
optimization_patience = 3
optimization_factor = 0.1

load_best_params = True    # Load the best parameters found by the Bayesian optimization from the best_params.txt file
model_save_name = "CAR-Net-Optimizer_large_dataset"     # Name of the model to save
checkpoint_path = f"D:\\CTA data\\models\\{model_save_name}_checkpoint"

if __name__ == "__main__":


    if BayesianOptimization:
        learning_rate, optimizer_name, batch_size, smoothing, schedule_type = optimization(number_of_trials)
    elif load_best_params:
        with open("D:\\CTA data\\models\\best_params.txt", "r") as f:
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



    model = CARNet().to(device)
    model.apply(weights_init)
    criterion = mPD_loss_2()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if use_scheduler:
        if schedule_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        elif schedule_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        elif schedule_type == 'None':
            scheduler = None
        else:
            assert False, "Invalid scheduler type"
    else:
        scheduler = None
        schedule_type = 'None'

    print(f"Training with learning rate: {learning_rate}, optimizer: {optimizer_name}, batch size: {batch_size}",
          f"smoothing: {smoothing}, scheduler: {schedule_type}")
    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
    train_loader, val_loader, test_loader = create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                            batch_size=batch_size, shuffle_train=True)

    trained_model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader,
                                                          num_epochs=number_of_epochs,
                                                          model_save_name=model_save_name,
                                                          checkpoint_path=checkpoint_path,
                                                          new_learning_rate_factor=None,
                                                          scheduler=scheduler,
                                                          smoothing=smoothing,
                                                          scheduler_type=schedule_type)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    test_loss = test_model(trained_model, criterion, test_loader)



