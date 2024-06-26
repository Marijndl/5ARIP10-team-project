# 5ARIP10-team-project
<img src="../logo_transparent.png" alt="Logo" height="75"/>

#### How to generate the dataset

The origin dataset is [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT), A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-Computed-Tomo. This dataset contains about 1000 3D CTA images.

**Step 1**: Download ImageCAS dataset and [3D Slicer](https://www.slicer.org/) (necessary for the next step)

**Step 2**: Extract the centerlines of the arteries from the CTA images, sample 350 points per segment by default. The 3D and 2D data are Cartesian coordinates (x,y,z) of the sample points, and the projected 2D data is 3D data with the z-axis component removed. 

+ Switch to the directory where you download the 3D slicer

  example (Windows CLI):
  `cd C:\AppData\Local\slicer.org\Slicer 5.6.1`

+ Run the extract_centerline.py in 3D slicer to generate centerlines, with arguments: the path of the downloaded CTA data, the path to save the output (extracted centerlines)

  example (Windows CLI):

  `Slicer.exe --python-script "C:\data_creation\extract_centerline.py" --no-splash --no-main-window 'd:/CTA data/1-1000' 'D:\\CTA data\\Segments'`


**Step 3**: Smooth the dataset

+ Run Interp_segments.py to interpolate the segments with Bspline to smooth the data. Change the directory to your own directory

**Step 4:** Create the deformed 2D dataset and the offset list

+ Run Deformed_data_creation.py, with the argument of the interpolated dataset directory

**Step 5:** Create spherical coordinates from Cartesian coordinates data, and load the 2 types of 3D and 2D files into PyTorch tensor, saved separately in 4 files: origin_2D_interp_353.pt, origin_3D_interp_353.pt, shape_2D_interp_353.pt, shape_3D_interp_353.pt

+ run data_tensor.py, with arguments: the path of deformed 2D dataset and the path of interpolated 3D dataset