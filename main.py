import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import CARNet
import numpy as np
import matplotlib.pyplot as plt

def convert_to_spherical(coordinates: np.array) -> (np.array, np.array):
    """
        Convert Cartesian coordinates to spherical coordinates.

        Parameters:
        - coordinates (np.array): An array containing Cartesian coordinates of points in three-dimensional space.

        Returns:
        - origin_tensor (np.array): The original point in Cartesian coordinates.
        - spherical_coordinates (np.array): An array containing spherical coordinates (r, theta, phi) for each point.
          - r (float): Radial distance from the origin.
          - theta (float): Polar angle, in radians, measured from the positive z-axis.
          - phi (float): Azimuthal angle, in radians, measured from the positive x-axis in the xy-plane.
    """
    spherical_coordinates = []
    origin_tensor = coordinates[0,:]

    for i in range(coordinates.shape[0]-1):
        [x,y,z] = coordinates[i+1,:] - coordinates[i,:]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/(np.sqrt(x**2+y**2)))
        spherical_coordinates.append([r, theta, phi])

    return origin_tensor, np.array(spherical_coordinates)


# Load 3D data
data_3d = np.genfromtxt('Segments_renamed/Segments_renamed/SegmentPoints_0001_00.csv', delimiter=',', skip_header=1)
coordinates_3d = data_3d[:, 1:4]  # Adjust based on your CSV's column layout

# Load 2D data
data_2d = np.genfromtxt('Segments_deformed_2/Segments_deformed_2/SegmentPoints_0001_00_def2D.csv', delimiter=',', skip_header=1)
d_data = data_2d[:, :3]  # Assuming first two columns after header are x, y coordinates
# Append a column of zeros to make it "3D" for processing
coordinates_2d = np.hstack((d_data, np.zeros((d_data.shape[0], 1))))

model = CARNet()

origin_3d, spherical_3d = convert_to_spherical(coordinates_3d)
origin_2d, spherical_2d = convert_to_spherical(coordinates_2d)

x3d_origin = torch.tensor(origin_3d, dtype=torch.float).view(1, 3, 1)  # Now shape is [1, 3, 1]
x2d_origin = torch.tensor(origin_2d, dtype=torch.float).view(1, 3, 1)  # Now shape is [1, 3, 1]

# Adjusting x3d_shape and x2d_shape to have shape (1, N, 3)
x3d_shape = torch.tensor(spherical_3d, dtype=torch.float).unsqueeze(0).transpose(1, 2)  # Now shape is [1, 349, 3]
x2d_shape = torch.tensor(spherical_2d, dtype=torch.float).unsqueeze(0).transpose(1, 2)


# Set the model to evaluation mode
model.eval()

# Pass the data through the model
with torch.no_grad():
    deformation_field = model(x3d_origin, x3d_shape, x2d_origin, x2d_shape)

# Print the shape of the output to understand its dimensions
print("Output shape:", deformation_field)

def convert_back_with_deformation_field(origin_tensor: np.array, spherical_coordinates: np.array, deformation_field: np.array) -> np.array:
    """
    Convert spherical coordinates back to Cartesian coordinates.

    Parameters:
    - origin_tensor (np.array): The original point in Cartesian coordinates.
    - spherical_coordinates (np.array): An array containing spherical coordinates (r, theta, phi) for each point.
      - r (float): Radial distance from the origin.
      - theta (float): Polar angle, in radians, measured from the positive z-axis.
      - phi (float): Azimuthal angle, in radians, measured from the positive x-axis in the xy-plane.

    Returns:
    - cartesian_coordinates (np.array): An array containing Cartesian coordinates of points in three-dimensional space.
    """
    cartesian_coordinates = []
    cartesian_coordinates.append(origin_tensor)

    matrix = torch.ones((349, 349))
    matrix = torch.triu(matrix)
    for i in range(spherical_coordinates.shape[0]):
        [r, theta, phi] = spherical_coordinates[i]
        x = r * torch.sin(theta + deformation_field[0, 0]) * torch.sin(phi + deformation_field[0, 1])            
        y = r * torch.sin(theta + deformation_field[0, 0]) * torch.cos(phi + deformation_field[0, 1])
        z = r * torch.cos(theta + deformation_field[0, 0])
        
        x = torch.add(torch.matmul(x, matrix), cartesian_coordinates[i][0][0])
        y = torch.add(torch.matmul(y, matrix), cartesian_coordinates[i][0][1])
        z = torch.add(torch.matmul(z, matrix), cartesian_coordinates[i][0][2])
        

        cartesian_coordinates.append(torch.stack((x,y,z)))

    return cartesian_coordinates[1]

reconstructed_coordinates = convert_back_with_deformation_field(x3d_origin, x3d_shape, deformation_field)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates from the input array
                              
x = coordinates_3d[:, 0]
y = coordinates_3d[:, 1]
z = coordinates_3d[:, 2]

x_r = reconstructed_coordinates[0, :]
y_r = reconstructed_coordinates[1, :]
z_r = reconstructed_coordinates[2, :]


x_2D = d_data[:, 0]
y_2D = d_data[:, 1]

# Plot the line connecting the points
ax.plot(x, y, z,)
ax.plot(x_r, y_r, z_r)
ax.plot(x_2D, y_2D)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(['Original centerline segment', 'Deformed centerline segment',  '2D loaded from csv file'])
plt.show()
