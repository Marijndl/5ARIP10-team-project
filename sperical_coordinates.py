import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipympl
mpl.use('Qt5Agg')


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

def convert_back(origin_tensor: np.array, spherical_coordinates: np.array) -> np.array:
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

    for i in range(spherical_coordinates.shape[0]):
        [r, theta, phi] = spherical_coordinates[i]
        current = cartesian_coordinates[i] + [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)]
        cartesian_coordinates.append(current)

    return np.array(cartesian_coordinates)

## Testing and validation plotting:

data = np.genfromtxt(f"D:\\CTA data\\Segments\\SegmentPoints_55_4.csv", delimiter=",")
coordinates = data[1:,1:4]
origin_tensor, spherical_coordinates = convert_to_spherical(coordinates)

reconstructed_coorinates = convert_back(origin_tensor, spherical_coordinates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates from the input array
x = coordinates[:, 0]
y = coordinates[:, 1]
z = coordinates[:, 2]

x_r = reconstructed_coorinates[:, 0]
y_r = reconstructed_coorinates[:, 1]
z_r = reconstructed_coorinates[:, 2]

# Plot the line connecting the points
ax.plot(x, y, z, marker='o', linestyle='-')
ax.plot(x_r, y_r, z_r)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()