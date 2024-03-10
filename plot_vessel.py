import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import pyvista as pv

cta_volume = nib.load('C:\\Users\\20203226\\Documents\\CTA data\\1-200\\1.img.nii.gz').get_fdata()
cta_vessels = nib.load('C:\\Users\\20203226\\Documents\\CTA data\\1-200\\1.label.nii.gz').get_fdata()

plt.imshow(cta_volume[:,:,3])
cta_3d = pv.wrap(cta_volume)
vessel_3d = pv.wrap(cta_vessels)


pl = pv.Plotter()
pl.add_volume(cta_3d)
pl.add_volume(vessel_3d)
pl.show()

# cta_3d.plot(volume=True) # Volume render
# vessel_3d.plot(volume=True) # Volume render


