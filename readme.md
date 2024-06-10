# 5ARIP10-team-project
### General Introduction
This project is a real-time 3D/2D coronary artery registration algorithm that transforms deformed 2D X-Ray artery image into matching 3D CT artery image. The model is based on the paper [CAR-Net: A Deep Learning-Based Deformation Model for 3D/2D Coronary Artery Registration](https://pubmed.ncbi.nlm.nih.gov/35436189/).

The model is deep learning based and uses UNet structure and dual-branch training. The dataset used for training is [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT), A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-Computed-Tomo. This dataset contains about 1000 3D CTA images.
### Project Structure

**main.py:**
The main.py file trains the CARNet model using 3D and 2D data with custom loss functions and optimization techniques, including data loading, model initialization, training with checkpoints, and learning rate scheduling.

**model.py:**
The model.py file defines the architecture of the CARNet model, including dual-branch processing for 2D and 3D inputs and a UNet backbone for generating a deformation field, featuring various layers and modules for downsampling, upsampling, and feature extraction.

**data_loader.py:**
The data_loader.py file defines classes for loading and preprocessing 2D and 3D data(conversion to spherical coordinates), and includes functions for splitting the dataset into training, validation, and test sets with DataLoaders.

**spherical_coordinates.py:** The spherical_coordinates.py file provides functions for converting between Cartesian and spherical coordinates, and for projecting 3D data to 2D. It includes methods for transforming coordinates to and from spherical form and for testing these transformations with example data.

**helper_function.py:** The helper_function.py file contains utility functions for saving, loading, and managing model checkpoints during the training process of the CARNet model. It includes functions to decide whether to overwrite existing models, save model states along with training parameters and statistics, and load model states from checkpoints.

**hyperparameter_optimization.py:** The hyperparameter_optimization.py file performs hyperparameter optimization for training the CARNet model using Optuna for Bayesian optimization. It defines the evaluation and objective functions for the optimization process and includes code to run and save the optimization results.

**inference_time.py:** The inference_time.py file measures the inference time of the CARNet model on a GPU by running a series of forward passes with dummy input data. It includes GPU warm-up, repeated timing of inference, and calculation of average inference time and standard deviation, followed by plotting the timing results.

**registration_demo.py:** The registration_demo.py file evaluates the CARNet model's performance on a test dataset by computing the mean projection distance (mPD) between deformed 3D points and original 2D points. It includes functions for loading the model and data, running the evaluation, and plotting results to visualize the deformations and original alignments.


### How to run the model
#### 1.  Download project

Clone the project using git clone:` git clone git@github.com:Marijndl/5ARIP10-team-project.git`

Create a directory 'CTA data' to store the generated data pt file, and create a directory 'models' under 'CTA data', to store the model.

#### 2.Create a New Environment


Open your terminal and install dependencies using the following command:

```sh
pip install -r requirements.txt
```

#### 3. Run the project
Run the project with command: `python main.py`