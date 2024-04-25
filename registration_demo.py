from torch.utils.data import DataLoader
from data_loaders import *
from model import CARNet

mpl.use('Qt5Agg')

def convert_to_projection(origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
    deformed = torch.tensor([])
    original = torch.tensor([])

    spherical_3D = spherical_3D.clone()

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

# Load the model
model = CARNet()
model.load_state_dict(torch.load("D:\\CTA data\\models\\CAR-Net-256-defor.pth"))

# Load dataset
train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Run model
data_iter = iter(train_loader)
sample = next(data_iter)
output = model(sample['origin_3D'], sample['shape_3D'], sample['origin_2D'], sample['shape_2D'])

# Convert back to cartesian
deformed, original = convert_to_projection(sample['origin_3D'], sample['shape_3D'], sample['origin_2D'],
                                           sample['shape_2D'], output.detach())

deformed = deformed.detach().numpy()
original = original.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

idx = 0

# Extract x, y, z coordinates from the input array
org_x = original[idx, 0, :]
org_y = original[idx, 1, :]
# org_z = original[idx, 2, :]
org_z = np.zeros(original[idx, 2, :].shape) # To prevent the small non-zero values to influence the plot

def_x = deformed[idx, 0, :]
def_y = deformed[idx, 1, :]
def_z = deformed[idx, 2, :]

# Plot the line connecting the points
ax.plot(org_x, org_y, org_z)
ax.plot(def_x, def_y, def_z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(['Original centerline segment', 'Deformed centerline segment'])
plt.show()
