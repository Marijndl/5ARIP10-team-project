import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
mpl.use('Qt5Agg')


def convert_to_spherical(coordinates: np.array):
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

    # return origin_tensor, np.array(spherical_coordinates)
    return torch.from_numpy(origin_tensor).float(), torch.from_numpy(np.array(spherical_coordinates)).float()

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

def convert_back_tensors(origin_tensor: torch.tensor, spherical_coordinates: torch.tensor) -> torch.tensor:
    """
    Convert spherical coordinates back to Cartesian coordinates.

    Parameters:
    - origin_tensor (torch.Tensor): The original point in Cartesian coordinates.
    - spherical_coordinates (torch.Tensor): An array containing spherical coordinates (r, theta, phi) for each point.
        - r (float): Radial distance from the origin.
        - theta (float): Polar angle, in radians, measured from the positive z-axis.
        - phi (float): Azimuthal angle, in radians, measured from the positive x-axis in the xy-plane.

    Returns:
    - torch.Tensor: An array containing Cartesian coordinates of points in three-dimensional space.
    """
    cartesian_coordinates = origin_tensor.clone().detach()

    for i in range(spherical_coordinates.shape[1]):
        r = spherical_coordinates[0, i]
        theta = spherical_coordinates[1, i]
        phi = spherical_coordinates[2, i]

        current = torch.add(cartesian_coordinates[:, i].unsqueeze(dim=1), torch.tensor([[r * torch.sin(theta) * torch.cos(phi)], [r * torch.sin(theta) * torch.sin(phi)], [r * torch.cos(theta)]]))
        cartesian_coordinates = torch.cat((cartesian_coordinates, current), dim=1)

    return cartesian_coordinates

def convert_to_projection_old(origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
    deformed = torch.tensor([])
    original = torch.tensor([])

    # Add deformation to 3D line
    spherical_3D[:, 1:, :] += deformation_field

    for idx in range(deformation_field.shape[0]):
        # Convert back to cartesian
        cartesian_2D = convert_back(torch.transpose(origin_2D[idx], 0, 1).detach().numpy(),
                                    torch.transpose(spherical_2D[idx], 0, 1).detach().numpy()).squeeze()
        cartesian_3D = convert_back(torch.transpose(origin_3D[idx], 0, 1).detach().numpy(),
                                    torch.transpose(spherical_3D[idx], 0, 1).detach().numpy()).squeeze()

        # Project to 2D
        cartesian_3D[:, 2] = np.zeros(cartesian_3D.shape[0])

        original = torch.cat((original, torch.unsqueeze(torch.transpose(torch.from_numpy(cartesian_2D).float(), 0, 1), dim=0)), dim=0)
        deformed = torch.cat((deformed, torch.unsqueeze(torch.transpose(torch.from_numpy(cartesian_3D).float(), 0, 1), dim=0)), dim=0)

    return deformed, original

if __name__ == "__main__":
    ## Testing and validation plotting:

    data = np.genfromtxt(f".\\CTA data\\Segments\\SegmentPoints_55_4.csv", delimiter=",")
    coordinates = data[1:,1:4]
    coordinates[:,0] -= np.min(coordinates[:,0])
    coordinates[:,1] -= np.min(coordinates[:,1])
    coordinates[:,2] -= np.min(coordinates[:,2])

    data_2D = np.genfromtxt(f".\\CTA data\\Segments_deformed\\SegmentPoints_55_4_def2D.csv", delimiter=",")
    coordinates_2D = data_2D
    # f = interp1d(coordinates_2D[:,0], coordinates_2D[:,1])

    coordinates_2D[:, 0] -= np.min(coordinates_2D[:, 0])
    coordinates_2D[:, 1] -= np.min(coordinates_2D[:, 1])

    #Convert to spherical
    origin_tensor, spherical_coordinates = convert_to_spherical(coordinates)
    origin_tensor_2, spherical_coordinates_2 = convert_to_spherical(np.hstack((coordinates_2D, np.zeros((coordinates_2D.shape[0], 1)))))

    # Deformation
    offset = 0.5*np.pi
    samples = np.linspace(offset, offset+2*np.pi, num=349)
    deformation = np.sin(samples)
    spherical_coordinates[:, 1] += 0.10 * deformation
    spherical_coordinates[:, 2] += 0.10 * deformation

    # Convert back to cartesian
    reconstructed_coordinates = convert_back(origin_tensor, spherical_coordinates)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from the input array
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    x_r = reconstructed_coordinates[:, 0]
    y_r = reconstructed_coordinates[:, 1]
    z_r = reconstructed_coordinates[:, 2]

    x_flat = reconstructed_coordinates[:, 0]
    y_flat = reconstructed_coordinates[:, 1]

    x_2D = coordinates_2D[:, 0]
    y_2D = coordinates_2D[:, 1]

    # Plot the line connecting the points
    ax.plot(x, y, z,)
    ax.plot(x_r, y_r, z_r)
    ax.plot(x_flat, y_flat)
    # ax.plot(x_2D, y_2D, ":")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Original centerline segment', 'Deformed centerline segment', '2D projection deformed segment', '2D loaded from csv file'])
    plt.show()