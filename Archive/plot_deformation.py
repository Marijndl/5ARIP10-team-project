import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
mpl.use('Qt5Agg')
from scipy import interpolate
import re
from spherical_coordinates import *
import random

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

base_dir = "D:\\CTA data\\Segments bspline 353\\"
deform_dir = "D:\\CTA data\\Segments bspline\\"
i = 10

data = np.genfromtxt(f"D:\\CTA data\\Segments bspline 353\\SegmentPoints_0024_{str(i).zfill(2)}.csv", delimiter=",")

data[:, 0] -= np.min(data[:, 0])
data[:, 1] -= np.min(data[:, 1])
data[:, 2] -= np.min(data[:, 2])

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.plot(x, y, z, 'orange', label='Original centerline segment')

# Convert to spherical
origin_tensor, spherical_coordinates = convert_to_spherical(data)

# Deformation
offset = random.uniform(0, 2*np.pi)
samples = np.linspace(offset, offset + 2 * np.pi, num=352)
deformation = np.sin(samples)
spherical_coordinates[:, 1] += 0.50 * deformation
spherical_coordinates[:, 2] += 0.50 * deformation

reconstructed_coordinates = convert_back(origin_tensor, spherical_coordinates)

x_new = reconstructed_coordinates[:, 0]
y_new = reconstructed_coordinates[:, 1]
z_new = reconstructed_coordinates[:, 2]

ax.plot(x_new, y_new, z_new, '-c', label='Artificially deformed centerline segment')

ax.plot(x_new, y_new, 'g', label='2D projection deformed segment')


ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('Artificial deformations')
plt.legend(loc='best')
plt.show()