import random

from torch.utils.data import DataLoader
from data_loaders import *
from model import CARNet
from main import mPD_loss_2
import time
mpl.use('Qt5Agg')

device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 5e9 else "cpu")

# Load the model
checkpoint = torch.load("D:\\CTA data\\models\\CAR-Net-Optimizer_trial0_val.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model = CARNet().to(device)
loss = mPD_loss_2()

# Load the data
train_loader, val_loader, test_loader = create_datasets(CenterlineDatasetSpherical(base_dir="D:\\CTA data\\"), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=512, shuffle_train=False)

def evaluate_model(model, test_loader, loss, plot_outliers=False, std_threshold=10):
    model.eval()
    all_distances = []
    times = []
    for batch in test_loader:
        start = time.time()
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
        end = time.time()
        times.append(end - start)

        deformed_3D = deformed_cart_3D.clone().detach().cpu().numpy()
        original_3D = original_cart_3D.clone().detach().cpu().numpy()
        original_2D = original_cart_2D.clone().detach().cpu().numpy()
        # Remove z component
        difference = abs(deformed_3D - original_2D)
        difference = difference[:, :2, :]

        # pythagorean theorem
        distances = np.mean(np.sqrt(np.sum(difference ** 2, axis=1)), axis=1)
        all_distances.extend(distances)
        standard_deviation = np.std(all_distances)
        threshold = np.mean(all_distances) + std_threshold * standard_deviation
        for idx, distance in enumerate(distances):
            if distance > threshold:
                print(f"Sample {idx} has a distance of {distance} which is above the threshold of {threshold}")
                if plot_outliers == True:
                    plot_3D_centerline(original_3D, deformed_3D, original_2D, distances, idx)
        #plot random sample

    plot_3D_centerline(original_3D, deformed_3D, original_2D, distances, random.randint(0, len(distances)-1))
    print(f"Number of samples: {len(all_distances)}")
    mPD = np.mean(all_distances)
    std_mPD = np.std(all_distances)
    print(f"Mean time per sample: {np.mean(times):.2f}")
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
    #def_z = np.zeros(original_2D[idx, 2, :].shape)

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

    ax.legend(['Original 2D centerline segment', 'Deformed 3D centerline segment', 'Original 3D centerline segment'])
    plt.show()

evaluate_model(model, val_loader, loss, plot_outliers=False, std_threshold=10)