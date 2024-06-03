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


def save_model(model, epoch, batch_size, learning_rate, smoothing, optimizer, train_loss, val_loss, best_val_loss, model_save_name, scheduler=None, scheduler_type='None'):
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