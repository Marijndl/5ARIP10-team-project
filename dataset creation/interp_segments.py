import matplotlib as mpl
import numpy as np

mpl.use('Qt5Agg')
from scipy import interpolate
import os

base_dir = "D:\\CTA data\\Segments original\\"
interp_dir = "D:\\CTA data\\Segments bspline\\"
file_list = os.listdir(base_dir)

def interpolate_segment(base_dir, interp_dir, segment):
    data = np.genfromtxt(os.path.join(base_dir, segment), delimiter=",")[1:, 1:4]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Create a parametric variable 't' based on the number of data points
    t = np.arange(x.shape[0])

    # Number of points on the interpolated curve
    num_points = 350

    # Perform B-spline interpolation in 3D
    tck_x = interpolate.splrep(t, x, s=0, k=3)
    tck_y = interpolate.splrep(t, y, s=0, k=3)
    tck_z = interpolate.splrep(t, z, s=0, k=3)

    # Create a new parameter range
    t_new = np.linspace(t[0], t[-1], num_points)

    # Evaluate the B-spline curves at the new parameter values
    x_new = interpolate.BSpline(*tck_x)(t_new).reshape(num_points, 1)
    y_new = interpolate.BSpline(*tck_y)(t_new).reshape(num_points, 1)
    z_new = interpolate.BSpline(*tck_z)(t_new).reshape(num_points, 1)

    # Save the new data:
    header = "X, Y, Z"
    np.savetxt(os.path.join(interp_dir, segment), np.concatenate((x_new, y_new, z_new), axis=1),
               delimiter=",", header=header)

# Loop over all the segment files
for segment in file_list:
    try:
        interpolate_segment(base_dir, interp_dir, segment)
    except:
        print(f"{segment} failed to interpolate")
        pass

