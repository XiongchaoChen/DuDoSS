import random
import numpy as np
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
import os
import h5py

# (1) Training dataset
class LVTrain(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])
        # self.data_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.data_dir) for f in filenames if os.path.splitext(f)[1] == '.h5'])

        self.vol_fv_sinogram_all = []
        self.vol_lv_sinogram_all = []
        self.vol_lv_sinogram_inter_all = []
        self.vol_mask_sinogram_all = []
        self.vol_mask_crop_all = []
        self.vol_fv_recon_all = []
        self.vol_lv_recon_all = []

        # Load all images
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_fv_sinogram = f['fv_sinogram'][...].transpose(2,1,0)
                vol_lv_sinogram = f['lv_sinogram'][...].transpose(2,1,0)
                vol_lv_sinogram_inter = f['lv_sinogram_inter'][...].transpose(2,1,0)
                vol_mask_sinogram = f['mask_sinogram'][...].transpose(2,1,0)
                vol_mask_crop = f['mask_crop'][...].transpose(2,1,0)       # [60,32,64] - [64,32,60]
                vol_fv_recon = f['fv_recon'][...].transpose(2,1,0)
                vol_lv_recon = f['lv_recon'][...].transpose(2,1,0)        # [32,64,64] - [64,64,32]

            # Creating random indexes for patching (sinogram [64, 32, 60])
            X_template = vol_fv_sinogram
            indexes = get_random_patch_indexes(data=X_template, patch_size= [64,32,60], num_patches= 1, padding='VALID')

            # Using index for patching
            X_patches = get_patches_from_indexes(image=vol_fv_sinogram, indexes=indexes, patch_size= [64,32,60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_fv_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_sinogram, indexes=indexes, patch_size= [64,32,60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_sinogram_inter, indexes=indexes, patch_size=[64,32,60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_sinogram_inter_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_mask_sinogram, indexes=indexes, patch_size= [64,32,60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_mask_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_mask_crop, indexes=indexes, patch_size= [64,32,60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_mask_crop_all.append(X_patches)


            # Creating random indexes for patching (recon [64, 64, 32])
            X_template = vol_fv_recon
            indexes = get_random_patch_indexes(data=X_template, patch_size= [64, 64, 32], num_patches=1, padding='VALID')

            # Using index for patching
            X_patches = get_patches_from_indexes(image=vol_fv_recon, indexes=indexes, patch_size=[64, 64, 32], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_fv_recon_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_recon, indexes=indexes, patch_size=[64, 64, 32], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_recon_all.append(X_patches)

        self.vol_fv_sinogram_all = np.concatenate(self.vol_fv_sinogram_all, 0)
        self.vol_lv_sinogram_all = np.concatenate(self.vol_lv_sinogram_all, 0)
        self.vol_lv_sinogram_inter_all = np.concatenate(self.vol_lv_sinogram_inter_all, 0)
        self.vol_mask_sinogram_all = np.concatenate(self.vol_mask_sinogram_all, 0)
        self.vol_mask_crop_all = np.concatenate(self.vol_mask_crop_all, 0)
        self.vol_fv_recon_all = np.concatenate(self.vol_fv_recon_all, 0)
        self.vol_lv_recon_all = np.concatenate(self.vol_lv_recon_all, 0)

    def __getitem__(self, index):
        vol_fv_sinogram = self.vol_fv_sinogram_all[index, ...]
        vol_lv_sinogram = self.vol_lv_sinogram_all[index, ...]
        vol_lv_sinogram_inter = self.vol_lv_sinogram_inter_all[index, ...]
        vol_mask_sinogram = self.vol_mask_sinogram_all[index, ...]
        vol_mask_crop = self.vol_mask_crop_all[index, ...]
        vol_fv_recon = self.vol_fv_recon_all[index, ...]
        vol_lv_recon = self.vol_lv_recon_all[index, ...]

        vol_fv_sinogram = torch.from_numpy(vol_fv_sinogram.copy())
        vol_lv_sinogram = torch.from_numpy(vol_lv_sinogram.copy())
        vol_lv_sinogram_inter = torch.from_numpy(vol_lv_sinogram_inter.copy())
        vol_mask_sinogram = torch.from_numpy(vol_mask_sinogram.copy())
        vol_mask_crop = torch.from_numpy(vol_mask_crop.copy())
        vol_fv_recon = torch.from_numpy(vol_fv_recon.copy())
        vol_lv_recon = torch.from_numpy(vol_lv_recon.copy())

        return {'fv_sinogram': vol_fv_sinogram,
                'lv_sinogram': vol_lv_sinogram,
                'lv_sinogram_inter': vol_lv_sinogram_inter,
                'mask_sinogram':vol_mask_sinogram,
                'mask_crop':vol_mask_crop,
                'fv_recon': vol_fv_recon,
                'lv_recon': vol_lv_recon}

    def __len__(self):
        return len(self.data_files)


# (2) Validation dataset
class LVVal(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'valid')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])
        # self.data_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.data_dir) for f in filenames if os.path.splitext(f)[1] == '.h5'])

        self.vol_fv_sinogram_all = []
        self.vol_lv_sinogram_all = []
        self.vol_lv_sinogram_inter_all = []
        self.vol_mask_sinogram_all = []
        self.vol_mask_crop_all = []
        self.vol_fv_recon_all = []
        self.vol_lv_recon_all = []

        # Load all images
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_fv_sinogram = f['fv_sinogram'][...].transpose(2, 1, 0)
                vol_lv_sinogram = f['lv_sinogram'][...].transpose(2, 1, 0)
                vol_lv_sinogram_inter = f['lv_sinogram_inter'][...].transpose(2, 1, 0)
                vol_mask_sinogram = f['mask_sinogram'][...].transpose(2, 1, 0)
                vol_mask_crop = f['mask_crop'][...].transpose(2, 1, 0)  # [60,32,64] - [64,32,60]
                vol_fv_recon = f['fv_recon'][...].transpose(2, 1, 0)
                vol_lv_recon = f['lv_recon'][...].transpose(2, 1, 0)  # [32,64,64] - [64,64,32]

            # Creating random indexes for patching (sinogram [64, 32, 60])
            X_template = vol_fv_sinogram
            indexes = get_random_patch_indexes(data=X_template, patch_size=[64, 32, 60], num_patches=1, padding='VALID')

            # Using index for patching
            X_patches = get_patches_from_indexes(image=vol_fv_sinogram, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_fv_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_sinogram, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_sinogram_inter, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_sinogram_inter_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_mask_sinogram, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_mask_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_mask_crop, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_mask_crop_all.append(X_patches)

            # Creating random indexes for patching (recon [64, 64, 32])
            X_template = vol_fv_recon
            indexes = get_random_patch_indexes(data=X_template, patch_size=[64, 64, 32], num_patches=1, padding='VALID')

            # Using index for patching
            X_patches = get_patches_from_indexes(image=vol_fv_recon, indexes=indexes, patch_size=[64, 64, 32], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_fv_recon_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_recon, indexes=indexes, patch_size=[64, 64, 32], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_recon_all.append(X_patches)

        self.vol_fv_sinogram_all = np.concatenate(self.vol_fv_sinogram_all, 0)
        self.vol_lv_sinogram_all = np.concatenate(self.vol_lv_sinogram_all, 0)
        self.vol_lv_sinogram_inter_all = np.concatenate(self.vol_lv_sinogram_inter_all, 0)
        self.vol_mask_sinogram_all = np.concatenate(self.vol_mask_sinogram_all, 0)
        self.vol_mask_crop_all = np.concatenate(self.vol_mask_crop_all, 0)
        self.vol_fv_recon_all = np.concatenate(self.vol_fv_recon_all, 0)
        self.vol_lv_recon_all = np.concatenate(self.vol_lv_recon_all, 0)


    def __getitem__(self, index):
        vol_fv_sinogram = self.vol_fv_sinogram_all[index, ...]
        vol_lv_sinogram = self.vol_lv_sinogram_all[index, ...]
        vol_lv_sinogram_inter = self.vol_lv_sinogram_inter_all[index, ...]
        vol_mask_sinogram = self.vol_mask_sinogram_all[index, ...]
        vol_mask_crop = self.vol_mask_crop_all[index, ...]
        vol_fv_recon = self.vol_fv_recon_all[index, ...]
        vol_lv_recon = self.vol_lv_recon_all[index, ...]

        vol_fv_sinogram = torch.from_numpy(vol_fv_sinogram.copy())
        vol_lv_sinogram = torch.from_numpy(vol_lv_sinogram.copy())
        vol_lv_sinogram_inter = torch.from_numpy(vol_lv_sinogram_inter.copy())
        vol_mask_sinogram = torch.from_numpy(vol_mask_sinogram.copy())
        vol_mask_crop = torch.from_numpy(vol_mask_crop.copy())
        vol_fv_recon = torch.from_numpy(vol_fv_recon.copy())
        vol_lv_recon = torch.from_numpy(vol_lv_recon.copy())

        return {'fv_sinogram': vol_fv_sinogram,
                'lv_sinogram': vol_lv_sinogram,
                'lv_sinogram_inter': vol_lv_sinogram_inter,
                'mask_sinogram': vol_mask_sinogram,
                'mask_crop': vol_mask_crop,
                'fv_recon': vol_fv_recon,
                'lv_recon': vol_lv_recon}

    def __len__(self):
        return len(self.data_files)


# (3) Testing dataset
class LVTest(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'test')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])
        # self.data_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.data_dir) for f in filenames if os.path.splitext(f)[1] == '.h5'])

        self.vol_fv_sinogram_all = []
        self.vol_lv_sinogram_all = []
        self.vol_lv_sinogram_inter_all = []
        self.vol_mask_sinogram_all = []
        self.vol_mask_crop_all = []
        self.vol_fv_recon_all = []
        self.vol_lv_recon_all = []

        # Load all images
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_fv_sinogram = f['fv_sinogram'][...].transpose(2, 1, 0)
                vol_lv_sinogram = f['lv_sinogram'][...].transpose(2, 1, 0)
                vol_lv_sinogram_inter = f['lv_sinogram_inter'][...].transpose(2, 1, 0)
                vol_mask_sinogram = f['mask_sinogram'][...].transpose(2, 1, 0)
                vol_mask_crop = f['mask_crop'][...].transpose(2, 1, 0)  # [60,32,64] - [64,32,60]
                vol_fv_recon = f['fv_recon'][...].transpose(2, 1, 0)
                vol_lv_recon = f['lv_recon'][...].transpose(2, 1, 0)  # [32,64,64] - [64,64,32]

            # Creating random indexes for patching (sinogram [64, 32, 60])
            X_template = vol_fv_sinogram
            indexes = get_random_patch_indexes(data=X_template, patch_size=[64, 32, 60], num_patches=1, padding='VALID')

            # Using index for patching
            X_patches = get_patches_from_indexes(image=vol_fv_sinogram, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_fv_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_sinogram, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_sinogram_inter, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_sinogram_inter_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_mask_sinogram, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_mask_sinogram_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_mask_crop, indexes=indexes, patch_size=[64, 32, 60], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_mask_crop_all.append(X_patches)

            # Creating random indexes for patching (recon [64, 64, 32])
            X_template = vol_fv_recon
            indexes = get_random_patch_indexes(data=X_template, patch_size=[64, 64, 32], num_patches=1, padding='VALID')

            # Using index for patching
            X_patches = get_patches_from_indexes(image=vol_fv_recon, indexes=indexes, patch_size=[64, 64, 32], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_fv_recon_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_lv_recon, indexes=indexes, patch_size=[64, 64, 32], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.vol_lv_recon_all.append(X_patches)

        self.vol_fv_sinogram_all = np.concatenate(self.vol_fv_sinogram_all, 0)
        self.vol_lv_sinogram_all = np.concatenate(self.vol_lv_sinogram_all, 0)
        self.vol_lv_sinogram_inter_all = np.concatenate(self.vol_lv_sinogram_inter_all, 0)
        self.vol_mask_sinogram_all = np.concatenate(self.vol_mask_sinogram_all, 0)
        self.vol_mask_crop_all = np.concatenate(self.vol_mask_crop_all, 0)
        self.vol_fv_recon_all = np.concatenate(self.vol_fv_recon_all, 0)
        self.vol_lv_recon_all = np.concatenate(self.vol_lv_recon_all, 0)


    def __getitem__(self, index):
        vol_fv_sinogram = self.vol_fv_sinogram_all[index, ...]
        vol_lv_sinogram = self.vol_lv_sinogram_all[index, ...]
        vol_lv_sinogram_inter = self.vol_lv_sinogram_inter_all[index, ...]
        vol_mask_sinogram = self.vol_mask_sinogram_all[index, ...]
        vol_mask_crop = self.vol_mask_crop_all[index, ...]
        vol_fv_recon = self.vol_fv_recon_all[index, ...]
        vol_lv_recon = self.vol_lv_recon_all[index, ...]

        vol_fv_sinogram = torch.from_numpy(vol_fv_sinogram.copy())
        vol_lv_sinogram = torch.from_numpy(vol_lv_sinogram.copy())
        vol_lv_sinogram_inter = torch.from_numpy(vol_lv_sinogram_inter.copy())
        vol_mask_sinogram = torch.from_numpy(vol_mask_sinogram.copy())
        vol_mask_crop = torch.from_numpy(vol_mask_crop.copy())
        vol_fv_recon = torch.from_numpy(vol_fv_recon.copy())
        vol_lv_recon = torch.from_numpy(vol_lv_recon.copy())

        return {'fv_sinogram': vol_fv_sinogram,
                'lv_sinogram': vol_lv_sinogram,
                'lv_sinogram_inter': vol_lv_sinogram_inter,
                'mask_sinogram': vol_mask_sinogram,
                'mask_crop': vol_mask_crop,
                'fv_recon': vol_fv_recon,
                'lv_recon': vol_lv_recon}

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    pass