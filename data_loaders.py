import os
from torch.utils.data import Dataset
from spherical_coordinates import *

offset_list = np.genfromtxt("D:\\CTA data\\Offset_deformations.txt", delimiter=",")

class CenterlineDataset(Dataset):
    def __init__(self, data_dir_2D, data_dir_3D, transform=None):
        self.data_dir_2D = data_dir_2D
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.file_list_2D = os.listdir(data_dir_2D)
        self.file_list_3D = os.listdir(data_dir_3D)

    def __len__(self):
        return len(self.file_list_2D)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load from file
        data_2d = np.genfromtxt(os.path.join(self.data_dir_2D, self.file_list_2D[idx]), delimiter=",")
        data_3d = np.genfromtxt(os.path.join(self.data_dir_3D, self.file_list_3D[idx]), delimiter=",")[1:, 1:4]

        # Add row of zeros
        data_2d = np.hstack((data_2d, np.zeros((data_2d.shape[0], 1))))

        # Normalize 3D
        data_3d[:, 0] -= np.min(data_3d[:, 0])
        data_3d[:, 1] -= np.min(data_3d[:, 1])
        data_3d[:, 2] -= np.min(data_3d[:, 2])

        # Convert to spherical:
        origin_2D, shape_2D = convert_to_spherical(data_2d)
        origin_3D, shape_3D = convert_to_spherical(data_3d)

        origin_2D = torch.reshape(origin_2D, (3, 1)).float()
        origin_3D = torch.reshape(origin_3D, (3, 1)).float()

        shape_2D = torch.transpose(shape_2D, 0, 1).float()
        shape_3D = torch.transpose(shape_3D, 0, 1).float()

        sample = {'origin_2D': origin_2D, 'origin_3D': origin_3D, 'shape_2D': shape_2D, 'shape_3D': shape_3D,
                  'offset': offset_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class CenterlineDatasetSpherical(Dataset):
    def __init__(self, base_dir, transform=None):
        self.origin_2D = torch.load(os.path.join(base_dir, "origin_2D.pt"))
        self.origin_3D = torch.load(os.path.join(base_dir, "origin_3D.pt"))
        self.shape_2D = torch.load(os.path.join(base_dir, "shape_2D.pt"))
        self.shape_3D = torch.load(os.path.join(base_dir, "shape_3D.pt"))
        self.transform = transform

    def __len__(self):
        return self.shape_2D.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'origin_2D': self.origin_2D[idx], 'origin_3D': self.origin_3D[idx], 'shape_2D': self.shape_2D[idx],
                  'shape_3D': self.shape_3D[idx], 'offset': offset_list[idx]}
        # sample = (self.origin_2D[idx], self.origin_3D[idx], self.shape_2D[idx], self.shape_3D[idx], offset_list[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample
