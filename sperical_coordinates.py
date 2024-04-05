import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    ## Testing and validation plotting:

    data = np.genfromtxt(f"D:\\CTA data\\Segments\\SegmentPoints_55_4.csv", delimiter=",")
    coordinates = data[1:,1:4]
    coordinates[:,0] -= np.min(coordinates[:,0])
    coordinates[:,1] -= np.min(coordinates[:,1])
    coordinates[:,2] -= np.min(coordinates[:,2])

    data_2D = np.genfromtxt(f"D:\\CTA data\\Segments_deformed\\SegmentPoints_55_4_def2D.csv", delimiter=",")
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