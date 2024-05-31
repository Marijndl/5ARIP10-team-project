import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
mpl.use('Qt5Agg')
from scipy import interpolate
import re

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(19):
    data = np.genfromtxt(f"D:\\CTA data\\Segments original\\SegmentPoints_0001_{str(i).zfill(2)}.csv", delimiter=",")[1:,1:4]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Create a parametric variable 't' based on the number of data points
    t = np.arange(x.shape[0])

    # Number of points on the interpolated curve
    num_points = 150

    # Perform B-spline interpolation in 3D
    tck_x = interpolate.splrep(t, x, s=0, k=3)
    tck_y = interpolate.splrep(t, y, s=0, k=3)
    tck_z = interpolate.splrep(t, z, s=0, k=3)

    # Create a new parameter range
    t_new = np.linspace(t[0], t[-1], num_points)

    # Evaluate the B-spline curves at the new parameter values
    x_new = interpolate.BSpline(*tck_x)(t_new)
    y_new = interpolate.BSpline(*tck_y)(t_new)
    z_new = interpolate.BSpline(*tck_z)(t_new)

    # Plot the original and interpolated data points in 3D

    # ax.scatter(x, y, z, c='r', marker='o', label='Original Data')
    ax.plot(x, y, z, '--g', label='Original Data')
    ax.plot(x_new, y_new, z_new, '-c', label='B-spline Interpolation')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('B-spline Interpolation in 3D')
plt.legend(loc='best')
plt.show()