import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
mpl.use('Qt5Agg')
from spherical_coordinates import *
import random

if __name__ == "__main__":
    # Define the directory where your files are located
    directory = ".\\CTA data\\Segments renamed\\"

    file_names = os.listdir(directory)

    offset_list = []

    for file in file_names:

        # Load cartesian coordinates
        data = np.genfromtxt(os.path.join(directory, file), delimiter=",")
        coordinates = data[1:, 1:4]

        # Normalize
        coordinates[:, 0] -= np.min(coordinates[:, 0])
        coordinates[:, 1] -= np.min(coordinates[:, 1])
        coordinates[:, 2] -= np.min(coordinates[:, 2])

        # Convert to spherical
        origin_tensor, spherical_coordinates = convert_to_spherical(coordinates)

        # Deformation
        offset = random.uniform(0, 2*np.pi)
        offset_list.append(offset)
        samples = np.linspace(offset, offset + 2 * np.pi, num=349)
        deformation = np.sin(samples)
        spherical_coordinates[:, 1] += 0.10 * deformation
        spherical_coordinates[:, 2] += 0.10 * deformation

        print(file + " Offset: " + str(offset))

        # Convert back to cartesian
        reconstructed_coordinates = convert_back(origin_tensor, spherical_coordinates)

        # Save to CSV with header
        header = "X, Y"
        np.savetxt(f".\\CTA data\\Segments_deformed_3\\{file[:-4]}_def2D.csv", reconstructed_coordinates[:,:2], delimiter=",", header=header)

    # Save the offsets to a .txt file:
    with open(f".\\CTA data\\Offset_deformations.txt", 'w') as file:
        for item in offset_list:
            file.write(str(item) + '\n')

