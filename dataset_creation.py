import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import pyvista as pv
import math

def projection_to_2D(input_tensor: np.array, alpha: float, beta: float, dxc: float, dxd: float) -> np.array:

    #Convert angles to radians:
    alpha = math.radians(alpha)
    beta = math.radians(beta)

    #Projection
    p_base = np.sqrt(1 + np.tan(alpha)**2 + np.tan(beta)**2)
    p_x = -dxc*np.tan(alpha) / p_base
    p_y = dxc / p_base
    p_z = -dxc*np.tan(beta) / p_base
    p = np.array([p_x, p_y, p_z])

    #Axes of X-ray source coordinate system
    z_axis = -p / dxc
    x_axis = np.array([p_y / np.sqrt(p_x**2 + p_y**2), -p_x / np.sqrt(p_x**2 + p_y**2), 0])
    y_base = np.sqrt((p_x**2 + p_y**2)**2 + p_x**2*p_z**2 + p_y**2*p_z**2)
    y_axis = np.array([-p_x*p_z / y_base, -p_x*p_z / y_base, -p_x*p_z / y_base])

    coordinates = []
    for z in range(input_tensor.shape[2]):
        for x in range(input_tensor.shape[0]):
            for y in range(input_tensor.shape[1]):
                if input_tensor[x,y,z] != 0:
                    v_x = x * 0.37695312
                    v_y = y * 0.37695312
                    v_z = z * 0.5

                    #Convert point in volume to point in X-ray source
                    r_x = np.dot(np.array([v_x - p_x, 0, 0]), x_axis)
                    r_y = np.dot(np.array([0, v_y - p_y, 0]), y_axis)
                    r_z = np.dot(np.array([0, 0, v_z - p_z]), z_axis)   #(v_z - p_x) * z_axis
                    coordinates.append([r_x * dxd / r_z, r_y * dxd / r_z])

    coordinates = np.array(coordinates)
    coordinates[:,0] -= np.min(coordinates[:,0])
    coordinates[:,1] -= np.min(coordinates[:,1])
    # coordinates = coordinates/np.max(coordinates) * 512
    return coordinates

if __name__ == "__main__":
    cta_vessels = nib.load('C:\\Users\\20203226\\Documents\\CTA data\\1-200\\1.label.nii.gz').get_fdata()
    coords = projection_to_2D(cta_vessels, -30, 25, 38, 56)

    fig, ax = plt.subplots()
    ax.scatter(coords[:,0], coords[:,1], s=2)
    # ax.set_ylim([0, 512])
    # ax.set_xlim([0, 512])
    plt.show()
    pass
