#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import matplotlib.pyplot as plt
import os
import argparse
from monai.transforms import NormalizeIntensity
from tqdm import tqdm
import pickle
from glob import glob
import shutil
from pathlib import Path

# third party
import numpy as np
import nibabel as nib
import torch
from nnunet.lib.training_utils import read_config_video, build_flow_model_successive, read_config, build_2d_model
from nnunet.network_architecture.integration import SpatialTransformer
from nnunet.lib.utils import ConvBlocks2DGroup
from nnunet.network_architecture.Optical_flow_model_successive import ModelWrap

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
import matplotlib
import kornia
import ants

from nnunet.training.network_training.nnMTLTrainerV2Raft import ModelWrap


def plot_coords_deformable(sampling_locations, attention_weights, images):
        """sampling_locations: T, B, H, W, self.heads * self.points, 2,
        attention_weights: T, B, H, W, self.heads * self.points,
        images: T, B, C, H, W"""

        T, B, C, H, W = images.shape
        _, _, H2, W2, h, P, _ = sampling_locations.shape

        #print(coords[0, 0, 0])

        init_x_coord = H2 // 2
        init_y_coord = W2 // 2

        scaling_ratio = 8

        sampling_locations = sampling_locations[:, 0, init_x_coord, init_y_coord, :, :] # T, h, P, 2
        attention_weights = attention_weights[:, 0, init_x_coord, init_y_coord, :] # T, h, P

        if H2 % 2 != 0:
            init_x_coord = init_x_coord + 0.5
            init_y_coord = init_y_coord + 0.5

        coords_xy = (sampling_locations + 1) * (H / 2)

        matplotlib.use('QtAgg')
        fig, ax = plt.subplots(1, T, figsize=(5,1))

        attention_weights = attention_weights.detach().cpu()
        attention_weights = attention_weights.permute(1, 0, 2).contiguous()

        coords_xy = coords_xy.detach().cpu()
        coords_xy = coords_xy.permute(1, 0, 2, 3).contiguous()
        
        ax[-1].imshow(images[-1, 0, 0].detach().cpu(), cmap='gray')
        ax[-1].scatter([init_x_coord * scaling_ratio], [init_y_coord * scaling_ratio], color='green', marker='x')
        ax[-1].axis('off')
        for j in range(len(attention_weights)):
            vmax = attention_weights[j].max() # T, P
            vmin = attention_weights[j].min() # T, P
            for i in range(T-1):
                ax[i].imshow(images[i, 0, 0].cpu(), cmap='gray')
                ax[i].scatter(coords_xy[j, i, :, 0].detach().cpu(), coords_xy[j, i, :, 1].detach().cpu(), cmap='hot', c=attention_weights[j, i], vmin=vmin, vmax=vmax, marker='o', s=2)
                ax[i].axis('off')
        
        fig.tight_layout()

        plt.show()


def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)



def inference_iterative(path_list_gz, newpath_flow, newpath_registered, model, image_size):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    x_list = []
    gt_list = []
    mask_list = []
    for path_gz in path_list_gz:
        data = np.load(path_gz)
        x = data[0][None, None]
        gt = data[1][None, None]
        mask = data[2:-1][None]

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(mask[0, 0, :, :, 0], cmap='hot')
        #ax[1].imshow(mask[0, 1, :, :, 0], cmap='hot')
        #ax[2].imshow(mask[0, 2, :, :, 0], cmap='hot')
        #plt.show()

        x = torch.from_numpy(x).to(device).float()
        gt = torch.from_numpy(gt).to(device).float()
        mask = torch.from_numpy(mask).to(device).float()

        #x, fixed_affine = vxm.py.utils.load_volfile(path_gz, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        #gt = vxm.py.utils.load_volfile(path_gt, add_batch_axis=True, add_feat_axis=add_feat_axis)
        
        x_list.append(x)
        gt_list.append(gt)
        mask_list.append(mask)
    
    x = torch.stack(x_list, dim=0)
    gt = torch.stack(gt_list, dim=0)
    mask = torch.stack(mask_list, dim=0)

    fixed_path = path_list_gz[0]

    flow_list_all = []
    moved_list_all = []
    img_list_all = []
    for d in range(x.shape[-1]):
        current_fixed = x[0, :, :, :, :, d]
        out_list = []
        out_list_flow = []
        out_list_img = []
        for t in range(len(path_list_gz)):
            moving_path = path_list_gz[t]
            if fixed_path == moving_path:
                continue
            filename = os.path.basename(moving_path)

            current_moving = x[t, :, :, :, :, d]
            moving_seg = gt[t, :, :, :, :, d]
        
            x_in = torch.stack([current_fixed, current_moving], dim=0)
            x_in = NormalizeIntensity()(x_in)

            flow_list = model(x_in)
            warp = flow_list[-1]

            ed_target = torch.nn.functional.one_hot(moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
            moved = motion_estimation(flow=warp, original=ed_target, mode='bilinear')
            #moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W

            out_list.append(moved[0])
            out_list_flow.append(warp[0].permute(1, 2, 0).contiguous())
            out_list_img.append(current_moving[0, 0])
        
        moved = torch.stack(out_list, dim=0).detach().cpu().numpy() # T-1, C, H, W
        flow = torch.stack(out_list_flow, dim=0).detach().cpu().numpy()
        img_sequence = torch.stack(out_list_img, dim=0).detach().cpu().numpy()

        flow_list_all.append(flow)
        moved_list_all.append(moved)
        img_list_all.append(img_sequence)
    
    moved = np.stack(moved_list_all, axis=-1) # T-1, C, H, W, D
    img = np.stack(img_list_all, axis=-1) # T-1, H, W, D
    flow = np.stack(flow_list_all, axis=-2) # T-1, H, W, D, 2

    for t in range(len(moved)):
        # save moved image
        moving_path = path_list_gz[t + 1]

        filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path = os.path.join(newpath_registered, filename)
        np.savez(save_path, seg=moved[t].squeeze())

        flow_filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow[t].squeeze(), img=img[t].squeeze())



# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--test_or_val', required=True, help='Whether this is testing set or validation_set')
parser.add_argument('--dataset', required=True, help='dataset (ACDC or Lib)')
parser.add_argument('--dirpath', required=True, help='output directory path')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

newpath_flow_forward = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Forward_flow')
delete_if_exist(newpath_flow_forward)
os.makedirs(newpath_flow_forward)

newpath_flow_backward = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Backward_flow')
delete_if_exist(newpath_flow_backward)
os.makedirs(newpath_flow_backward)

newpath_registered_forward = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Registered_forward')
delete_if_exist(newpath_registered_forward)
os.makedirs(newpath_registered_forward)

newpath_registered_backward = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Registered_backward')
delete_if_exist(newpath_registered_backward)
os.makedirs(newpath_registered_backward)

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

add_feat_axis = not args.multichannel

config = read_config_video(os.path.join(os.path.dirname(args.model), 'config.yaml'))

config['fine_tuning'] = False
config['mamba'] = False

if args.dataset == 'ACDC':
    with open(os.path.join('splits', 'ACDC', 'splits_final.pkl'), 'rb') as f:
        data = pickle.load(f)
        training_patients = data[0]['train']
        validation_patients = data[0]['val']
    
    if args.test_or_val == 'test':
        path_list = glob(os.path.join('voxelmorph_ACDC_2D_testing', '*.gz'))
        path_list_gt = glob(os.path.join('voxelmorph_ACDC_gt_2D_testing', '*.gz'))
        path_list_pkl = glob(os.path.join('voxelmorph_ACDC_2D_testing', '*.pkl'))
    elif args.test_or_val == 'val':
        path_list = glob(os.path.join('voxelmorph_ACDC_2D', '*.gz'))
        path_list_gt = glob(os.path.join('voxelmorph_ACDC_gt_2D', '*.gz'))
        path_list_pkl = glob(os.path.join('voxelmorph_ACDC_2D', '*.pkl'))
    image_size = 128
elif args.dataset == 'Lib':
    if args.test_or_val == 'test':

        with open(os.path.join('splits', 'Lib', 'test', 'splits_final.pkl'), 'rb') as f:
            data = pickle.load(f)
            validation_patients = data[0]['val']

        path_list = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_testing_mask", '*.npy'))
        #path_list_pkl = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_testing_mask", '*.pkl'))

    elif args.test_or_val == 'val':

        with open(os.path.join('splits', 'Lib', 'val', 'splits_final.pkl'), 'rb') as f:
            data = pickle.load(f)
            validation_patients = data[0]['val']

        path_list = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_training_mask", '*.npy'))
    path_list_pkl = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\custom_lib_t_4", '**', '*.pkl'), recursive=True)
    image_size = 192

# load and set up model
# load the model

pretrained_network = None

model = ModelWrap(do_ds=False)

model.load_state_dict(torch.load(args.model)['state_dict'])
model = model.cuda()
model.eval()

validation_patients = sorted(list(set([x.split('_')[0] for x in validation_patients])))

path_list = sorted([x for x in path_list if os.path.basename(x).split('_')[0] in validation_patients])
path_list_pkl = sorted([x for x in path_list_pkl if os.path.basename(os.path.dirname(x)) in validation_patients])

assert len(path_list) == len(path_list_pkl)

patient_list = sorted(list(set([os.path.basename(x).split('_')[0] for x in path_list])))

all_patient_paths = []
all_patient_paths_pkl = []
for patient in patient_list:
    patient_files = []
    patient_files_pkl = []
    for (path, pkl_path) in zip(path_list, path_list_pkl):
        if patient in path:
            patient_files.append(path)
        if patient in pkl_path:
            patient_files_pkl.append(pkl_path)
    all_patient_paths.append(sorted(patient_files))
    all_patient_paths_pkl.append(sorted(patient_files_pkl))


for (path_list_gz, path_list_pkl) in tqdm(zip(all_patient_paths, all_patient_paths_pkl), total=len(all_patient_paths)):

    with open(path_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        ed_number = np.rint(data['ed_number']).astype(int)
        es_number = np.rint(data['es_number']).astype(int)

    path_list_gz = np.array(path_list_gz)
    frame_indices = np.arange(len(path_list_gz))
    after = frame_indices >= ed_number
    before = frame_indices < ed_number
    path_list_gz = np.concatenate([path_list_gz[after], path_list_gz[before]])

    assert int(os.path.basename(path_list_gz[0]).split('frame')[-1][:2]) == ed_number + 1

    with torch.no_grad():
        inference_iterative(path_list_gz, newpath_flow_backward, newpath_registered_backward, model, image_size)

    
