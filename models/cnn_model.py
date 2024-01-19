import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import time

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, mse, get_nonlinearity, nmse, nmae
from skimage.metrics import structural_similarity as ssim
from utils.data_patch_util import *
import pdb
import scipy.io as scio


class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []
        self.lr_G1 = opts.lr_G1
        self.lr_G2 = opts.lr_G2


        # Network
        self.net_G1 = get_generator(opts.net_G, opts, ic=1)  # image domain
        self.net_G2 = get_generator(opts.net_G, opts, ic=1)  # projection domain
        self.networks.append(self.net_G1)
        self.networks.append(self.net_G2)

        # Loss Name
        self.loss_names += ['loss_G1']
        self.loss_names += ['loss_G2']

        # Optimizer
        self.optimizer_G1 = torch.optim.Adam(self.net_G1.parameters(),
                                             lr=opts.lr_G1,
                                             betas=(opts.beta1, opts.beta2),
                                             weight_decay=opts.weight_decay)
        self.optimizer_G2 = torch.optim.Adam(self.net_G2.parameters(),
                                             lr=opts.lr_G2,
                                             betas=(opts.beta1, opts.beta2),
                                             weight_decay=opts.weight_decay)
        self.optimizers.append(self.optimizer_G1)
        self.optimizers.append(self.optimizer_G2)

        # Loss function
        self.criterion = nn.L1Loss()

        # Options
        self.opts = opts


    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def system_matrix(self, batch_size):
        self.SM = torch.tensor(scio.loadmat('./sm/matTrans.mat')['matTrans'], device=self.device).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 3840, 4096]
        # self.SM = torch.tensor(scio.loadmat('./sm/matTrans.mat')['matTrans'], device=self.device).float().unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 1, 3840, 4096]

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.fv_sinogram = data['fv_sinogram'].to(self.device).float()    # torch.Size([4, 1, 64, 32, 60])
        self.lv_sinogram = data['lv_sinogram'].to(self.device).float()
        self.lv_sinogram_inter = data['lv_sinogram_inter'].to(self.device).float()
        self.mask_sinogram = data['mask_sinogram'].to(self.device).float()
        self.mask_crop = data['mask_crop'].to(self.device).float()
        self.fv_recon = data['fv_recon'].to(self.device).float()
        self.lv_recon = data['lv_recon'].to(self.device).float()   # torch.Size([4, 1, 64, 64, 32])

        self.sm_size = self.fv_sinogram.size(0)

        # save_nii(self.fv_recon[0,0,...].detach().cpu().numpy(), 'demo/fv_recon.nii')
        # save_nii(self.lv_recon[0, 0, ...].detach().cpu().numpy(), 'demo/lv_recon.nii')
        # save_nii(self.fv_sinogram[0, 0, ...].detach().cpu().numpy(), 'demo/fv_sinogram.nii')
        # save_nii(self.lv_sinogram[0, 0, ...].detach().cpu().numpy(), 'demo/lv_sinogram.nii')
        # save_nii(self.mask_sinogram[0, 0, ...].detach().cpu().numpy(), 'demo/mask_sinogram.nii')

        # self.fv_recon_flat = self.fv_recon.rot90(2, [2, 3]).permute(0,1,3,2,4).reshape(self.batch_size, 1, 64*64, 32)  # torch.Size([4, 1, 4096, 32])
        # self.Proj = torch.matmul(self.SM, self.fv_recon_flat).reshape(self.batch_size, 1,60,64,32).permute([0,1,3,4,2]) # torch.Size([4, 1, 64, 32, 60])


    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        SM = self.SM.repeat(self.sm_size, 1, 1, 1)  # [B, 1, 3840, 4096]

        self.inp = self.lv_recon
        self.inp.requires_grad_(True)                    # torch.Size([4, 1, 64, 64, 32])

        # (1) ----------- image domain: image prediction --------------
        self.fv_recon_pred = self.net_G1(self.inp)       # torch.Size([4, 1, 64, 64, 32])


        # (2) ----------- Forward Projection ---------------------
        self.fv_recon_pred_flat = self.fv_recon_pred.rot90(2, [2, 3]).permute(0, 1, 3, 2, 4).reshape(self.sm_size, 1, 64 * 64, 32)     # torch.Size([4, 1, 4096, 32])
        self.fp_pred = torch.matmul(SM, self.fv_recon_pred_flat).reshape(self.sm_size, 1, 60, 64, 32).permute([0, 1, 3, 4, 2])    # torch.Size([4, 1, 64, 32, 60])

        # (3) ------------ Replace Angle 1 + Prediction ---------------------
        self.fp_pred_replace = self.fp_pred * self.mask_sinogram + self.lv_sinogram * (1 - self.mask_sinogram)
        self.fv_sinogram_pred = self.net_G2(self.fp_pred_replace)
        self.fv_sinogram_pred_replace = self.fv_sinogram_pred * self.mask_sinogram + self.lv_sinogram * (1 - self.mask_sinogram)

        # # (3) ----------------- Concatenation -------------------
        # self.cat_sinogram = torch.cat([self.lv_sinogram_inter, self.fp_pred, self.mask_sinogram], 1)
        # self.fv_sinogram_pred = self.net_G2(self.cat_sinogram)
        # self.fv_sinogram_pred_replace = self.fv_sinogram_pred * self.mask_sinogram + self.lv_sinogram * (1 - self.mask_sinogram)

        # save_nii(self.fv_recon_pred[0, 0, ...].detach().cpu().numpy(), 'demo/fv_recon_pred.nii')
        # save_nii(self.fp_pred[0, 0, ...].detach().cpu().numpy(), 'demo/fp_pred.nii')
        # save_nii(self.fp_pred_replace[0, 0, ...].detach().cpu().numpy(), 'demo/fp_pred_replace.nii')
        # save_nii(self.fv_sinogram_pred[0, 0, ...].detach().cpu().numpy(), 'demo/fv_sinogram_pred.nii')
        # save_nii(self.fv_sinogram_pred_replace[0, 0, ...].detach().cpu().numpy(), 'demo/fv_sinogram_pred_replace.nii')
        # pdb.set_trace()



    def update(self):
        # Zero gradient
        self.optimizer_G1.zero_grad()
        self.optimizer_G2.zero_grad()

        # Calculate the two loss idependently
        loss_G1 = self.criterion(self.fv_recon_pred, self.fv_recon)
        loss_G2 = self.criterion(self.fv_sinogram_pred_replace, self.fv_sinogram)

        self.loss_G1 = loss_G1.item()
        self.loss_G2 = loss_G2.item()

        # Calculate the total loss
        loss_total = loss_G1 + loss_G2
        loss_total.backward()
        self.optimizer_G1.step()
        self.optimizer_G2.step()

    @property
    def loss_summary(self):
        message = ''
        message += ' loss1: {:.4e}, loss2: {:.4e}'.format(self.loss_G1, self.loss_G2)
        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        self.lr_G1 = self.optimizers[0].param_groups[0]['lr']
        self.lr_G2 = self.optimizers[1].param_groups[0]['lr']


    def save(self, filename, epoch, total_iter):
        state = {}
        state['net_G1'] = self.net_G1.module.state_dict()
        state['net_G2'] = self.net_G2.module.state_dict()
        state['opt_G1'] = self.optimizer_G1.state_dict()
        state['opt_G2'] = self.optimizer_G2.state_dict()
        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        self.net_G1.module.load_state_dict(checkpoint['net_G1'])
        self.net_G2.module.load_state_dict(checkpoint['net_G2'])
        if train:
            self.optimizer_G1.load_state_dict(checkpoint['opt_G1'])
            self.optimizer_G2.load_state_dict(checkpoint['opt_G2'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def evaluate(self, loader):
        val_bar = tqdm(loader)

        # For calculating metrics
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_mse = AverageMeter()
        avg_nmse = AverageMeter()
        avg_nmae = AverageMeter()

        for data in val_bar:
            self.set_input(data)
            self.forward()

            # Crop image
            mask = self.mask_crop.squeeze().cpu().numpy()  # [64,32,60]
            mask_range = []
            for n in range(mask.shape[1]):  # 32
                if mask[:, n, :].sum().item() != 0:
                    mask_range.append(n)

            self.fv_sinogram_pred_replace_crop = self.fv_sinogram_pred_replace[:,:,:,mask_range,:]
            self.fv_sinogram_crop              = self.fv_sinogram[:,:,:,mask_range,:]

            psnr_sinogram = psnr(self.fv_sinogram_pred_replace_crop, self.fv_sinogram_crop)
            mse_sinogram =   mse(self.fv_sinogram_pred_replace_crop, self.fv_sinogram_crop)
            ssim_sinogram = ssim(self.fv_sinogram_pred_replace_crop[0, 0, ...].cpu().numpy(), self.fv_sinogram_crop[0, 0, ...].cpu().numpy())
            nmse_sinogram = nmse(self.fv_sinogram_pred_replace_crop, self.fv_sinogram_crop)
            nmae_sinogram = nmae(self.fv_sinogram_pred_replace_crop, self.fv_sinogram_crop)

            avg_psnr.update(psnr_sinogram)
            avg_mse.update(mse_sinogram)
            avg_ssim.update(ssim_sinogram)
            avg_nmse.update(nmse_sinogram)
            avg_nmae.update(nmae_sinogram)

            message = 'NMSE: {:4f}, NMAE: {:4f}, PSNR: {:4f}, SSIM: {:4f}'.format(avg_nmse.avg, avg_nmae.avg, avg_psnr.avg, avg_ssim.avg)
            val_bar.set_description(desc=message)

        # Calculate the average metrics
        self.nmse_sinogram = avg_nmse.avg
        self.nmae_sinogram = avg_nmae.avg
        self.psnr_sinogram = avg_psnr.avg
        self.ssim_sinogram = avg_ssim.avg
        self.mse_sinogram = avg_mse.avg


    # --------------- Save the images ------------------------------
    def save_images(self, loader, folder):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Saving images ...')

        # --------------- Mkdir save folder -------------------
        if not os.path.exists(os.path.join(folder, 'fv_sinogram')):
            os.mkdir(os.path.join(folder, 'fv_sinogram'))
        if not os.path.exists(os.path.join(folder, 'lv_sinogram')):
            os.mkdir(os.path.join(folder, 'lv_sinogram'))
        if not os.path.exists(os.path.join(folder, 'lv_sinogram_inter')):
            os.mkdir(os.path.join(folder, 'lv_sinogram_inter'))

        if not os.path.exists(os.path.join(folder, 'fv_sinogram_pred')):
            os.mkdir(os.path.join(folder, 'fv_sinogram_pred'))
        if not os.path.exists(os.path.join(folder, 'fv_sinogram_pred_replace')):
            os.mkdir(os.path.join(folder, 'fv_sinogram_pred_replace'))
        if not os.path.exists(os.path.join(folder, 'fv_recon_pred')):
            os.mkdir(os.path.join(folder, 'fv_recon_pred'))

        # for consistency
        if not os.path.exists(os.path.join(folder, 'refine_sinogram')):
            os.mkdir(os.path.join(folder, 'refine_sinogram'))

        # Load data for each batch
        index = 0
        for data in val_bar:
            index += 1
            self.set_input(data)  # [batch_szie=1, 1, 64, 64, 64]
            self.forward()

            # Crop image
            mask = self.mask_crop.squeeze().cpu().numpy()  # [64,32,60]
            mask_range = []
            for n in range(mask.shape[1]):  # 32
                if mask[:, n, :].sum().item() != 0:
                    mask_range.append(n)

            # save images
            save_nii(self.fv_sinogram[:,:,:,mask_range,:].squeeze().cpu().numpy(), os.path.join(folder, 'fv_sinogram', 'fv_sinogram_' + str(index) + '.nii'))
            save_nii(self.lv_sinogram[:,:,:,mask_range,:].squeeze().cpu().numpy(), os.path.join(folder, 'lv_sinogram', 'lv_sinogram_' + str(index) + '.nii'))
            save_nii(self.lv_sinogram_inter[:,:,:,mask_range,:].squeeze().cpu().numpy(), os.path.join(folder, 'lv_sinogram_inter', 'lv_sinogram_inter_' + str(index) + '.nii'))

            save_nii(self.fv_sinogram_pred[:,:,:,mask_range,:].squeeze().cpu().numpy(), os.path.join(folder, 'fv_sinogram_pred', 'fv_sinogram_pred_' + str(index) + '.nii'))
            save_nii(self.fv_sinogram_pred_replace[:,:,:,mask_range,:].squeeze().cpu().numpy(), os.path.join(folder, 'fv_sinogram_pred_replace', 'fv_sinogram_pred_replace_' + str(index) + '.nii'))
            save_nii(self.fv_recon_pred[:,:,:,:,mask_range].squeeze().cpu().numpy(), os.path.join(folder, 'fv_recon_pred', 'fv_recon_pred_' + str(index) + '.nii'))

            # For consistency
            save_nii(self.fv_sinogram_pred_replace[:,:,:,mask_range,:].squeeze().cpu().numpy(), os.path.join(folder, 'refine_sinogram', 'refine_sinogram_' + str(index) + '.nii'))

