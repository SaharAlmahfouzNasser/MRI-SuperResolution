import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from torchvision import transforms as T


class MRIDataset(Dataset):
    def __init__(self, lr_path, hr_path):
        super().__init__()
        self.lr = nib.load(lr_path).get_fdata().transpose(3, 2, 1, 0)
        self.hr = nib.load(hr_path).get_fdata().transpose(3, 2, 1, 0)

    def __len__(self):
        return self.lr.shape[0]

    def __getitem__(self, i):
        lr_im = self.lr[i, :, :, :]
        hr_im = self.hr[i, :, :, :]

        return torch.from_numpy(lr_im), torch.from_numpy(hr_im)


class MRIDatasetNpy(Dataset):
    def __init__(self, lr_path, hr_path, lr_norm_path, hr_norm_path, norm=True):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.lrs = os.listdir(self.lr_path)
        self.hrs = os.listdir(self.hr_path)
        self.lr_norm = np.load(lr_norm_path)
        self.hr_norm = np.load(hr_norm_path)
        self.norm = norm

    def __len__(self):
        return len(self.lrs)

    def __getitem__(self, i):
        lr_im = np.load(os.path.join(self.lr_path, self.lrs[i]))
        hr_im = np.load(os.path.join(self.hr_path, self.hrs[i]))

        lr_im = lr_im[np.newaxis, :]
        hr_im = hr_im[np.newaxis, :]

        if self.norm:
            lr_im = (lr_im - self.lr_norm) / self.lr_norm
            hr_im = (hr_im - self.hr_norm) / self.hr_norm

        return torch.from_numpy(lr_im).float(), torch.from_numpy(hr_im).float()


class MRIDatasetNpySubmission(Dataset):
    def __init__(self, lr_path, lr_norm_path, norm=True):
        super().__init__()
        self.lr_path = lr_path
        self.lrs = os.listdir(self.lr_path)
        self.lr_norm = np.load(lr_norm_path)
        self.norm = norm

    def __len__(self):
        return len(self.lrs)

    def __getitem__(self, i):
        lr_im = np.load(os.path.join(self.lr_path, self.lrs[i]))

        lr_im = lr_im[np.newaxis, :]

        if self.norm:
            lr_im = (lr_im - self.lr_norm) / self.lr_norm

        return torch.from_numpy(lr_im).float(), self.lrs[i]
