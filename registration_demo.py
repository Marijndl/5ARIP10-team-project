from torch.utils.data import DataLoader
from data_loaders import *
from model import CARNet
from main import mPD_loss_2

mpl.use('Qt5Agg')

# Load the model
model = CARNet().to('cuda')
model.load_state_dict(torch.load("D:\\CTA data\\models\\CAR-Net-256-20.pth"))
loss = mPD_loss_2()

# Load dataset
train_dataset = CenterlineDatasetSpherical(base_dir="D:\\CTA data\\")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


device = 'cuda'
# Run model
data_iter = iter(train_loader)
sample = next(data_iter)
sample['origin_3D'].requires_grad_(True)
sample['shape_3D'].requires_grad_(True)
sample['origin_2D'].requires_grad_(True)
sample['shape_2D'].requires_grad_(True)
sample['origin_3D'].retain_grad()
sample['shape_3D'].retain_grad()
sample['origin_2D'].retain_grad()
sample['shape_2D'].retain_grad()
deformation_field = model(sample['origin_3D'].to(device), sample['shape_3D'].to(device), sample['origin_2D'].to(device), sample['shape_2D'].to(device))
deformation_field.requires_grad_(True)
deformation_field.retain_grad()


spherical_3D_deformed = sample['shape_3D'].clone().to('cuda')
spherical_3D_deformed[:, 1:, :] = torch.add(spherical_3D_deformed[:, 1:, :], deformation_field)

# Convert back to cartesian domain
deformed_cart_3D = loss.cartesian_tensor(sample['origin_3D'].to('cuda'), spherical_3D_deformed)
original_cart_3D = loss.cartesian_tensor(sample['origin_3D'].to('cuda'), sample['shape_3D'].to('cuda'))
original_cart_2D = loss.cartesian_tensor(sample['origin_2D'].to('cuda'), sample['shape_2D'].to('cuda'))

deformed_3D = deformed_cart_3D.clone().detach().cpu().numpy()
original_3D = original_cart_3D.clone().detach().cpu().numpy()
original_2D = original_cart_2D.clone().detach().cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

idx = 0

# Extract x, y, z coordinates from the input array
org_x = original_2D[idx, 0, :]
org_y = original_2D[idx, 1, :]
# org_z = original_2D[idx, 2, :]
org_z = np.zeros(original_2D[idx, 2, :].shape) # To prevent the small non-zero values to influence the plot

def_x = deformed_3D[idx, 0, :]
def_y = deformed_3D[idx, 1, :]
# def_z = deformed_3D[idx, 2, :]
def_z = np.zeros(original_2D[idx, 2, :].shape)

org_x_3 = original_3D[idx, 0, :]
org_y_3 = original_3D[idx, 1, :]
org_z_3 = original_3D[idx, 2, :]


# Plot the lines connecting the points
ax.plot(org_x, org_y, org_z)
ax.plot(def_x, def_y, def_z)
ax.plot(org_x_3, org_y_3, org_z_3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(['Original 2D centerline segment', 'Deformed 3D centerline segment', 'Original 3D centerline segment'])
plt.show()
