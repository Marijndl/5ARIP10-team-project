import numpy as np

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
    origin_tensor = coordinates[1,:]
    
    for i in range(coordinates.shape[0]-1):
        [x,y,z] = coordinates[i+1,:] - coordinates[i,:]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/(np.sqrt(x**2+y**2)))
        spherical_coordinates.append([r, theta, phi])

    return origin_tensor, np.array(spherical_coordinates)

data = np.genfromtxt(f"D:\\CTA data\\Segments\\SegmentPoints_1_0.csv", delimiter=",")
coordinates = data[1:,1:4]
origin_tensor, spherical_coordinates = convert_to_spherical(coordinates)
pass