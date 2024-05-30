from data_loaders import *
from main import mPD_loss_2
from model import CARNet
from torch.utils.data import DataLoader, random_split
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



train_loader, val_loader, test_loader = create_datasets(CenterlineDatasetSpherical(base_dir="D:\\CTA data\\"), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=256, shuffle_train=True)
model = CARNet().to(device)
model.load_state_dict(torch.load("D:\\CTA data\\models\\CAR-Net-256-20.pth"))


def test_output_statistics(test_loader):
    model.eval()
    mean_projection_distances = []

    with torch.no_grad():
        for input in test_loader:
            input['origin_3D'] = input['origin_3D'].to(device)
            input['shape_3D'] = input['shape_3D'].to(device)
            input['origin_2D'] = input['origin_2D'].to(device)
            input['shape_2D'] = input['shape_2D'].to(device)

            # Get the model outputs (deformation_field)
            deformation_field = model(input['origin_3D'], input['shape_3D'], input['origin_2D'], input['shape_2D'])

            # Deform the spherical 3D coordinates with the deformation field
            spherical_3D_deformed = input['shape_3D'].clone()
            spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], deformation_field)

            # Compute cartesian coordinates
            deformed_cart = mPD_loss_2().cartesian_tensor(input['origin_3D'], spherical_3D_deformed)
            original_cart = mPD_loss_2().cartesian_tensor(input['origin_2D'], input['shape_2D'])

            # Calculate mean projection distance for the current sample
            mPD = calculate_output_statistics(deformed_cart, original_cart)
            mean_projection_distances.append(mPD)

    # Calculate the mean and standard deviation of the mean projection distances
    mean_mPD = np.mean(mean_projection_distances)
    std_mPD = np.std(mean_projection_distances)

    return mean_mPD, std_mPD

def calculate_output_statistics(deformed_cart, original_cart):
    # Calculate the mean projection distance for a single output
    mPD = torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1))
    return mPD.item()



mean_mPD, std_mPD = test_output_statistics(test_loader)
print(f"Mean Projection Distance: {mean_mPD}")
print(f"Standard Deviation of Projection Distance: {std_mPD}")
