import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
mpl.use('Qt5Agg')
from scipy import interpolate
import re

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

base_dir = "D:\\CTA data\\Segments original\\"
interp_dir = "D:\\CTA data\\Segments bspline\\"
i = 20

data = np.genfromtxt(f"D:\\CTA data\\Segments original\\SegmentPoints_1000_{str(i).zfill(2)}.csv", delimiter=",")[1:,1:4]
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
ax.plot(x, y, z, '--r', label='Original Data')

data = np.genfromtxt(f"D:\\CTA data\\Segments bspline\\SegmentPoints_1000_{str(i).zfill(2)}.csv", delimiter=",")[1:, :]
x_new = data[:, 0]
y_new = data[:, 1]
z_new = data[:, 2]
ax.plot(x_new, y_new, z_new, '-c', label='B-spline Interpolation')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('B-spline Interpolation in 3D')
plt.legend(loc='best')
plt.show()