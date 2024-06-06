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
from main import *

#Pytorch settings
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 5e9 else "cpu")



BayesianOptimization = False # Set to True to perform Bayesian optimization
number_of_trials = 20
optimization_epochs = 15
optimization_step_size = 6
optimization_gamma = 0.1
optimization_patience = 3
optimization_factor = 0.2

def evaluate_model(model, test_loader, loss):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model: The trained model.
    - test_loader: DataLoader for the test set.
    - loss: The loss function.

    Returns:
    - mPD: Mean Projection Distance.
    - std_mPD: Standard deviation of the projection distance.
    """
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
    #plot_3D_centerline(original_3D, deformed_3D, original_2D, distances, -1)

    mPD = np.mean(all_distances)
    std_mPD = np.std(all_distances)
    # print(f"Mean time per sample: {np.mean(times):.2f}")
    print(f"Mean Projection Distance: {mPD:.2f}")
    print(f"Standard deviation: {std_mPD:.2f}")
    return mPD, std_mPD


def objective(trial):
    """
    Objective function for Bayesian optimization using Optuna.

    Parameters:
    - trial: An Optuna trial object.

    Returns:
    - mPd: Mean Projection Distance for the validation set.
    """
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
    name = f"CAR-Net-Optimizerv2_trial{trial.number}"

    dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\", load_all=False)
    train_loader, val_loader, test_loader = create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                            batch_size=batch_size, shuffle_train=True)
    print(f"Training with learning rate: {learning_rate}, optimizer: {optimizer_name}, batch size: {batch_size}, smoothing: {smoothing}, scheduler: {schedule_type}")
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader,batch_size, learning_rate,
                                                          num_epochs=optimization_epochs,
                                                          smoothing=smoothing,
                                                          scheduler=scheduler,
                                                          scheduler_type=schedule_type,
                                                          model_save_name=name,)
    validation_name = name + "_val"
    best_model = load_model(model, optimizer, f"D:\\CTA data\\models\\{validation_name}", scheduler)[0]
    best_model.eval()
    mPd, std_mPd = evaluate_model(best_model, val_loader, criterion)
    stop_training = False
    return mPd


def optimization(trails=30):
    """
    Perform Bayesian optimization using Optuna.

    Parameters:
    - trials: Number of trials to run.

    Returns:
    - Best hyperparameters found during optimization.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trails)


    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    # save study
    with open("D:\\CTA data\\models\\study.pkl", "wb") as f:
        pickle.dump(study, f)


    # with open("D:\\CTA data\\models\\study.pkl", "rb") as f:
    #     study = pickle.load(f)

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

if __name__ == "__main__":

    #Perform Bayesian hyperparameter tuning using Optuna
    learning_rate, optimizer_name, batch_size, smoothing, schedule_type = optimization(number_of_trials)