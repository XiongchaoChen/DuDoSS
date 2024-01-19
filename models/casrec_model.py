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


class CasRecModel(nn.Module):
    def __init__(self, opts):
        super(CasRecModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []
        self.lr = opts.lr

        # set default loss flags
        loss_flags = ["w_img_L1"]
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()

        self.opts = opts

        self.nc = opts.nc

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.lv_sinogram = data['lv_sinogram'].to(self.device).float()
        self.fv_sinogram = data['fv_sinogram'].to(self.device).float()
        self.mask_sinogram = data['mask_sinogram'].to(self.device).float()
        self.att_sinogram = data['att_sinogram'].to(self.device).float()

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        x = self.lv_sinogram
        if self.opts.use_att:
            x = torch.cat([x, self.att_sinogram], 1)
        x.requires_grad_(True)

        net = {}
        h = None
        for i in range(1, self.nc+1):
            x = x.contiguous()

            if i > 1:
                h = h.contiguous()

            net['r%d_pred' % i], h = self.net_G(x, h)  # output CNN sinogram
            net['r%d_pred_dc' % i] = self.mask_sinogram * self.lv_sinogram + (1 - self.mask_sinogram) * net['r%d_pred' % i]
            x = net['r%d_pred_dc' % i]

        self.refine_sinogram = net['r%d_pred_dc' % i]

    def update_G(self):
        self.optimizer_G.zero_grad()

        loss_G_L1 = self.criterion(self.refine_sinogram, self.fv_sinogram)
        self.loss_G_L1 = loss_G_L1.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        message += 'G_L1: {:.4e} '.format(self.loss_G_L1)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        self.lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):
        state = {}
        state['net_G'] = self.net_G.module.state_dict()
        state['opt_G'] = self.optimizer_G.state_dict()
        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        self.net_G.module.load_state_dict(checkpoint['net_G'])
        if train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def evaluate(self, loader):
        val_bar = tqdm(loader)

        # For calculating metrics
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_mse = AverageMeter()
        avg_nmse  = AverageMeter()
        avg_nmae = AverageMeter()

        for data in val_bar:
            self.set_input(data)
            self.forward()

            psnr_sinogram = psnr(self.refine_sinogram, self.fv_sinogram)
            mse_sinogram = mse(self.refine_sinogram, self.fv_sinogram)
            ssim_sinogram = ssim(self.refine_sinogram[0, 0, ...].cpu().numpy(), self.fv_sinogram[0, 0, ...].cpu().numpy())
            nmse_sinogram = nmse(self.refine_sinogram, self.fv_sinogram)
            nmae_sinogram = nmae(self.refine_sinogram, self.fv_sinogram)

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
        if not os.path.exists(os.path.join(folder, 'lv_sinogram')):
            os.mkdir(os.path.join(folder, 'lv_sinogram'))

        if not os.path.exists(os.path.join(folder, 'fv_sinogram')):
            os.mkdir(os.path.join(folder, 'fv_sinogram'))

        if not os.path.exists(os.path.join(folder, 'refine_sinogram')):
            os.mkdir(os.path.join(folder, 'refine_sinogram'))

        if not os.path.exists(os.path.join(folder, 'att_sinogram')):
            os.mkdir(os.path.join(folder, 'att_sinogram'))


        # Load data for each batch
        index = 0
        for data in val_bar:
            index += 1
            self.set_input(data)  # [batch_szie=1, 1, 64, 64, 64]
            self.forward()

            # save images
            save_nii(self.lv_sinogram.squeeze().cpu().numpy().transpose(2, 1, 0),      os.path.join(folder, 'lv_sinogram', 'lv_sinogram_' + str(index) + '.nii'))
            save_nii(self.fv_sinogram.squeeze().cpu().numpy().transpose(2, 1, 0),      os.path.join(folder, 'fv_sinogram', 'fv_sinogram_' + str(index) + '.nii'))
            save_nii(self.refine_sinogram.squeeze().cpu().numpy().transpose(2, 1, 0),  os.path.join(folder, 'refine_sinogram', 'refine_sinogram_' + str(index) + '.nii'))
            save_nii(self.att_sinogram.squeeze().cpu().numpy().transpose(2, 1, 0),     os.path.join(folder, 'att_sinogram', 'att_sinogram_' + str(index) + '.nii'))