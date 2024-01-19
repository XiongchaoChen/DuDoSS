import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv
import pdb
from utils.data_patch_util import *


parser = argparse.ArgumentParser(description='LVSPECT')

# model name
parser.add_argument('--experiment_name', type=str, default='experiment_train_casrec', help='give a experiment name before training')
parser.add_argument('--model_type', type=str, default='model_casrec', help='give a model name before training: model_svrhd / model_svrld / model_vm')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='../Data/Processed_Stress_LA_120degrees/', help='data root folder')
parser.add_argument('--dataset', type=str, default='LV', help='dataset name')

# network architectures, (discriminators e.g. cD, sD, are not used in the paper)
parser.add_argument('--net_G', type=str, default='scSERDUNet', help='generator network')
parser.add_argument('--nc', type=int, default=3, help='number of cascade')
parser.add_argument('--UNet_depth', type=int, default=3, help='network depth')
parser.add_argument('--UNet_filters', type=int, default=5, help='UNet filters/channels in the first layer, 1 to 2^6')
parser.add_argument('--DuRDN_filters', type=int, default=64, help='DuRDN filters, 64')

# normalization
parser.add_argument('--norm', type=str, default='None', help='Normalization for each convolution')  # 'BN' ,'IN', or 'None'

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='epoch')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--n_patch_train', type=int, default=1, help='number of patch to crop for training')
parser.add_argument('--patch_size_train', nargs='+', type=int, default=[60, 64, 64], help='randomly cropped patch size for train')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=5, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=1, help='save evaluation for every number of epochs')
parser.add_argument('--n_patch_test', type=int, default=1, help='number of patch to crop for evaluation')
parser.add_argument('--patch_size_test', nargs='+', type=int, default=[60, 64, 64], help='ordered cropped patch size for evaluation')
parser.add_argument('--n_patch_valid', type=int, default=1, help='number of patch to crop for evaluation')
parser.add_argument('--patch_size_valid', nargs='+', type=int, default=[60, 64, 64], help='ordered cropped patch size for evaluation')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_G1', type=float, default=1e-4, help='learning rate (chain 1)')
parser.add_argument('--lr_G2', type=float, default=5e-5, help='learning rate (chain 2)')


parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=30, help='step size for step scheduler ')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=5, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)
model.system_matrix(opts.batch_size)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {} \n'.format(num_param))

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume, train=True)

# Schedule: Learning rate decrease policy
model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set, test_set = get_datasets(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader  = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Json files
with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(options_str)

# New CSV files
with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:   # Write CSV, some metadata
    writer = csv.writer(f)
    writer.writerow(['epoch'] + model.loss_names)  # empty here

with open(os.path.join(output_directory, 'table_metrics_valid.csv'), 'w') as f:   # Write CSV, some metadata
    writer = csv.writer(f)
    writer.writerow(['epoch', 'nmse', 'nmae', 'ssim', 'psnr', 'mse'])  # empty here

with open(os.path.join(output_directory, 'table_metrics_test.csv'), 'w') as f:   # Write CSV, some metadata
    writer = csv.writer(f)
    writer.writerow(['epoch', 'nmse', 'nmae', 'ssim', 'psnr', 'mse'])  # empty here

# ########### Traing Loop ###############
for epoch in range(ep0, opts.n_epochs):
    train_bar = tqdm(train_loader)

    model.train()
    model.set_epoch(epoch)

    for it, data in enumerate(train_bar):
        total_iter += 1
        model.set_input(data)
        model.forward()
        model.update()
        # train_bar.set_description(desc='[Epoch {}]'.format(epoch) + model.loss_summary)
        train_bar.set_description(desc='[Epoch {}, lr1={:.3e}, lr2={:.3e}]'.format(model.curr_epoch, model.lr_G1, model.lr_G2) + model.loss_summary)  # progress bar description

    # Save loss per epoch
    with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:  # 'a' Progressively write
        writer = csv.writer(f)
        writer.writerow([epoch] + list(model.get_current_losses().values()))

    # Learning rate decay
    model.update_learning_rate()

    # save checkpoint
    if (epoch+1) % opts.snapshot_epochs == 0:
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)

    # evaluation
    if (epoch+1) % opts.eval_epochs == 0:
        print('Normal Evaluation ......')

        lv_name = os.path.join(image_directory, 'lv_{:03d}.png'.format(epoch))
        fv_name = os.path.join(image_directory, 'fv_{:03d}.png'.format(epoch))
        refined_name = os.path.join(image_directory, 'refined_{:03d}.png'.format(epoch))

        save_image(model.lv_sinogram.detach().transpose(2, 4).transpose(2, 0)[:, :, 0, :, :],       lv_name, normalize=True, scale_each=True, padding=5)  # [4,1,60,32,64] - [4,1,64,32,60]
        save_image(model.fv_sinogram.detach().transpose(2, 4).transpose(2, 0)[:, :, 0, :, :],       fv_name, normalize=True, scale_each=True, padding=5)
        save_image(model.fv_sinogram_pred.detach().transpose(2, 4).transpose(2, 0)[:, :, 0, :, :],  refined_name, normalize=True, scale_each=True, padding=5)

        # Validation
        model.eval()
        with torch.no_grad():
            model.evaluate(val_loader)

        with open(os.path.join(output_directory, 'table_metrics_valid.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, model.nmse_sinogram, model.nmae_sinogram, model.ssim_sinogram, model.psnr_sinogram, model.mse_sinogram])


        # Testing
        model.eval()
        with torch.no_grad():
            model.evaluate(test_loader)

        with open(os.path.join(output_directory, 'table_metrics_test.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, model.nmse_sinogram, model.nmae_sinogram, model.ssim_sinogram, model.psnr_sinogram, model.mse_sinogram])


