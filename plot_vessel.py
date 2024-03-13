import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.ndimage import binary_erosion

cta_volume = nib.load('C:\\Users\\20203226\\Documents\\CTA data\\1-200\\1.img.nii.gz').get_fdata()
cta_vessels = nib.load('C:\\Users\\20203226\\Documents\\CTA data\\1-200\\1.label.nii.gz').get_fdata()
cta_vessels = np.clip(cta_vessels, 0, 1)

center_line = binary_erosion(cta_vessels, iterations=3).astype(cta_vessels.dtype)


# cta_3d = pv.wrap(cta_volume)
vessel_3d = pv.wrap(center_line)

pl = pv.Plotter()
# pl.add_volume(cta_3d)
pl.add_volume(vessel_3d)
pl.show()

# cta_3d.plot(volume=True) # Volume render
# vessel_3d.plot(volume=True) # Volume render


