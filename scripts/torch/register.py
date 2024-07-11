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
from nnunet.network_architecture.Optical_flow_model_successive import ModelWrap
from nnunet.network_architecture.Optical_flow_model_simple import OpticalFlowModelSimple
from nnunet.lib.training_utils import build_flow_model_successive, build_flow_model_simple, read_config_video, build_flow_model_recursive_video, build_flow_model_video
from nnunet.network_architecture.integration import SpatialTransformer

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8


def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)



def inference_window_all_chunk_whole(path_list_gz, path_list_gt, newpath_flow, newpath_registered, newpath_seg, model, image_size, es_number, deep_supervision):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    es_idx = np.where([int(os.path.basename(path_list_gz[i]).split('frame')[-1][:2]) == es_number + 1 for i in range(len(path_list_gz))])[0][0]

    moving_seg_path = path_list_gt[0]
    moving_seg = vxm.py.utils.load_volfile(moving_seg_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)

    x_list = []
    for (path_gz, path_gt) in zip(path_list_gz, path_list_gt):
        x, fixed_affine = vxm.py.utils.load_volfile(path_gz, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        x = torch.from_numpy(x).to(device).float().permute(0, 4, 1, 2, 3)
        x_list.append(x)
    
    x = torch.stack(x_list, dim=0)

    out_list = []
    out_list_flow = []
    out_list_img = []
    for d in range(x.shape[-1]):
        current_x = x[:, :, :, :, :, d]
        current_moving_seg = moving_seg[:, :, :, :, d]
        ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()

        indices = torch.arange(1, len(current_x))
        chunk1 = indices[:es_idx-1]
        chunk2 = indices[es_idx-1:]
        chunk2 = torch.flip(chunk2, dims=[0])

        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])

        chunk_list_flow = []

        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
            
            chunk_x = current_x[chunk]
            chunk_x = NormalizeIntensity()(chunk_x)
                
            with torch.no_grad():
                out1, out2 = model(chunk_x, inference=False)
            flow_pred = out2['cumulated'].detach().to('cpu')
            del chunk_x
            torch.cuda.empty_cache()

            chunk_list_flow.append(flow_pred)

        nan_tensor_1 = torch.full(size=(len(chunk1),) + chunk_list_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_flow[0].to('cuda:0')
        nan_tensor_2 = chunk_list_flow[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        warp = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        assert len(warp) == len(current_x) - 1
        
        registered_list = []
        for t in range(len(warp)):
            registered = motion_estimation(flow=warp[t], original=ed_target, mode='bilinear')
            moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W
            registered_list.append(moved)
        moved = torch.stack(registered_list, dim=0)
        assert len(moved) == len(current_x) - 1

        out_list.append(moved[:, 0, 0])
        out_list_flow.append(warp[:, 0].permute(0, 2, 3, 1).contiguous())
        out_list_img.append(current_x[1:, 0, 0])
    
    moved = torch.stack(out_list, dim=3).detach().cpu().numpy() # T - 1, H, W, D
    flow = torch.stack(out_list_flow, dim=3).detach().cpu().numpy() # T - 1, H, W, D, 2
    img = torch.stack(out_list_img, dim=3).detach().cpu().numpy() # T - 1, H, W, D

    fixed_path = path_list_gz[0]
    filename = os.path.basename(fixed_path)
    save_path = os.path.join(newpath_seg, filename)

    for t in range(len(moved)):
        # save moved image
        fixed_path = path_list_gz[t + 1]

        filename = os.path.basename(fixed_path)
        save_path = os.path.join(newpath_registered, filename)
        vxm.py.utils.save_volfile(moved[t].squeeze(), save_path, fixed_affine)

        flow_filename = os.path.basename(fixed_path)[:-7] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow[t], img=img[t])



def inference_window_all_chunk(path_list_gz, path_list_gt, newpath_flow, newpath_registered, newpath_seg, model, image_size, es_number, deep_supervision):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    es_idx = np.where([int(os.path.basename(path_list_gz[i]).split('frame')[-1][:2]) == es_number + 1 for i in range(len(path_list_gz))])[0][0]

    moving_seg_path = path_list_gt[0]
    moving_seg = vxm.py.utils.load_volfile(moving_seg_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)

    x_list = []
    for (path_gz, path_gt) in zip(path_list_gz, path_list_gt):
        x, fixed_affine = vxm.py.utils.load_volfile(path_gz, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        x = torch.from_numpy(x).to(device).float().permute(0, 4, 1, 2, 3)
        x_list.append(x)
    
    x = torch.stack(x_list, dim=0)

    out_list = []
    out_list_flow = []
    out_list_img = []
    for d in range(x.shape[-1]):
        current_x = x[:, :, :, :, :, d]
        current_moving_seg = moving_seg[:, :, :, :, d]
        ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()

        indices = torch.arange(1, len(current_x))
        chunk1 = indices[:es_idx-1]
        chunk2 = indices[es_idx-1:]
        chunk2 = torch.flip(chunk2, dims=[0])

        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])

        chunk_list_flow = []

        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
            
            window_list = []
            for t in range(1, len(chunk)):
                current_chunk = chunk[:t+1]
                #if step > 1:
                #    current_chunk = torch.cat([current_chunk[::step], current_chunk[-1].view(-1,)], dim=0)
                assert len(current_chunk) >= 2

                chunk_x = current_x[current_chunk]
                chunk_x = NormalizeIntensity()(chunk_x)

                #diff1 = current_chunk[:-1] - current_chunk[-1]
                #diff2 = np.diff(current_chunk)
                #stacked_diff1 = np.stack([diff1, -diff1], axis=0) % len(current_x)
                #stacked_diff2 = np.stack([diff2, -diff2], axis=0) % len(current_x)
                #global_distances = stacked_diff1[~cn] / len(chunk)
                #local_distances = stacked_diff2[cn] / len(chunk)
                #global_distances = torch.from_numpy(global_distances)[:, None].float().to('cuda:0')
                #local_distances = torch.from_numpy(local_distances)[:, None].float().to('cuda:0')
                #assert len(local_distances) == len(chunk_x) - 1 == len(global_distances)
                #assert torch.all(global_distances[:-1] >= global_distances[1:])
                #assert global_distances[-1] == local_distances[-1]

                #fig, ax = plt.subplots(1, 10)
                #for i in range(10):
                #    ax[i].imshow(chunk_x[i, 0, 0].cpu(), cmap='gray')
                #plt.show()

                #max_memory_allocated = torch.cuda.max_memory_allocated(device=model.get_device())
                #print(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                
                with torch.no_grad():
                    out1, out2 = model(chunk_x, inference=False)
                if deep_supervision:
                    flow_pred = out2['flow'][0].detach().to('cpu')
                else:
                    flow_pred = out2['flow'].detach().to('cpu')
                window_list.append(flow_pred)
                del chunk_x
                torch.cuda.empty_cache()

            window_list = torch.stack(window_list, dim=0)
            chunk_list_flow.append(window_list)

        nan_tensor_1 = torch.full(size=(len(chunk1),) + chunk_list_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_flow[0].to('cuda:0')
        nan_tensor_2 = chunk_list_flow[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        warp = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        assert len(warp) == len(current_x) - 1
        
        registered_list = []
        for t in range(len(warp)):
            registered = motion_estimation(flow=warp[t], original=ed_target, mode='bilinear')
            moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W
            registered_list.append(moved)
        moved = torch.stack(registered_list, dim=0)
        assert len(moved) == len(current_x) - 1

        out_list.append(moved[:, 0, 0])
        out_list_flow.append(warp[:, 0].permute(0, 2, 3, 1).contiguous())
        out_list_img.append(current_x[1:, 0, 0])
    
    moved = torch.stack(out_list, dim=3).detach().cpu().numpy() # T - 1, H, W, D
    flow = torch.stack(out_list_flow, dim=3).detach().cpu().numpy() # T - 1, H, W, D, 2
    img = torch.stack(out_list_img, dim=3).detach().cpu().numpy() # T - 1, H, W, D

    fixed_path = path_list_gz[0]
    filename = os.path.basename(fixed_path)
    save_path = os.path.join(newpath_seg, filename)

    for t in range(len(moved)):
        # save moved image
        fixed_path = path_list_gz[t + 1]

        filename = os.path.basename(fixed_path)
        save_path = os.path.join(newpath_registered, filename)
        vxm.py.utils.save_volfile(moved[t].squeeze(), save_path, fixed_affine)

        flow_filename = os.path.basename(fixed_path)[:-7] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow[t], img=img[t])


def inference_iterative(path_list_gz, path_list_gt, newpath_flow, newpath_registered, model, image_size):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    moving_path = path_list_gz[0]
    moving_seg_path = path_list_gt[0]
    filename = os.path.basename(moving_path)

    moving = vxm.py.utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = vxm.py.utils.load_volfile(moving_seg_path, add_batch_axis=True, add_feat_axis=add_feat_axis)

    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)

    for fixed_path in path_list_gz:
        if fixed_path == moving_path:
            continue

        # load moving and fixed images
        
        fixed, fixed_affine = vxm.py.utils.load_volfile(fixed_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

        # set up tensors and permute
        input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

        out_list = []
        out_list_flow = []
        out_list_img = []
        for i in range(input_moving.shape[4]):
            current_moving = input_moving[:, :, :, :, i]
            current_moving_seg = moving_seg[:, :, :, :, i]
            current_fixed = input_fixed[:, :, :, :, i]

            x = torch.stack([current_moving, current_fixed], dim=0)
            x = NormalizeIntensity()(x)

            out1, out2 = model(x, inference=False)
            warp = out2['flow']

            ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
            registered = motion_estimation(flow=warp, original=ed_target, mode='bilinear')
            moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W

            out_list.append(moved[0, 0])
            out_list_flow.append(warp[0].permute(1, 2, 0).contiguous())
            out_list_img.append(current_fixed[0, 0])
        
        moved = torch.stack(out_list, dim=2).detach().cpu().numpy()
        flow = torch.stack(out_list_flow, dim=2).detach().cpu().numpy()
        img = torch.stack(out_list_img, dim=2).detach().cpu().numpy()

        # save moved image
        filename = os.path.basename(fixed_path)
        save_path = os.path.join(newpath_registered, filename)
        vxm.py.utils.save_volfile(moved.squeeze(), save_path, fixed_affine)

        flow_filename = os.path.basename(fixed_path)[:-7] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow, img=img)



def inference_iterative_warp(path_list_gz, path_list_gt, newpath_flow, newpath_registered, model, image_size):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    data = []
    for path in path_list_gz:
        arr, affine = vxm.py.utils.load_volfile(path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        x = torch.from_numpy(arr).to(device).float().permute(0, 4, 1, 2, 3)
        data.append(x)
    data = torch.stack(data, dim=0)
    data = data.permute(5, 0, 1, 2, 3, 4).contiguous()

    data_seg = []
    for path in path_list_gt:
        arr = vxm.py.utils.load_volfile(path, add_batch_axis=True, add_feat_axis=add_feat_axis)
        x = torch.from_numpy(arr).to(device).float().permute(0, 4, 1, 2, 3)
        data_seg.append(x)
    data_seg = torch.stack(data_seg, dim=0)
    data_seg = data_seg.permute(5, 0, 1, 2, 3, 4).contiguous()

    flow_list_all = []
    moved_list_all = []
    img_list_all = []
    for d in range(len(data)):
        current_data_depth_seg = data_seg[d]
        current_data_depth = data[d]
        ed_target = torch.nn.functional.one_hot(current_data_depth_seg[0, :, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        out_list_flow_depth = []
        out_list_img = []
        for t in range(len(current_data_depth) - 1):
            x = torch.stack([current_data_depth[t], current_data_depth[t + 1]], dim=0)
            x = NormalizeIntensity()(x)

            out1, out2 = model(x, inference=False)
            warp = out2['flow'] # B, C, H, W

            out_list_flow_depth.append(warp)
            out_list_img.append(current_data_depth[t + 1])
        
        flow_sequence = torch.stack(out_list_flow_depth, dim=0) # T, B, C, H, W
        img_sequence = torch.stack(out_list_img, dim=0).detach().cpu().numpy() # T, B, C, H, W
        assert len(flow_sequence) == len(img_sequence) == len(path_list_gz) - 1

        moved_list = []
        flow_list = []
        for t in range(len(flow_sequence)):
            flow_list.append(flow_sequence[t])
            ed_target = motion_estimation(flow=flow_sequence[t], original=ed_target, mode='bilinear')
            moved_list.append(torch.argmax(ed_target, dim=1, keepdim=True).int())
        
        moved = torch.stack(moved_list, dim=0).detach().cpu().numpy()
        flow = torch.stack(flow_list, dim=0).detach().cpu().numpy()
        flow_list_all.append(flow)
        moved_list_all.append(moved)
        img_list_all.append(img_sequence)
    
    moved = np.stack(moved_list_all, axis=-1) # T-1, 1, 1, H, W, D
    img = np.stack(img_list_all, axis=-1) # T-1, 1, 1, H, W, D
    flow = np.stack(flow_list_all, axis=-1) # T-1, 1, 2, H, W, D
    flow = flow.transpose(0, 1, 3, 4, 5, 2) # T-1, 1, H, W, D, 2

    for t in range(len(moved)):
        # save moved image
        fixed_path = path_list_gz[t + 1]

        filename = os.path.basename(fixed_path)
        save_path = os.path.join(newpath_registered, filename)
        vxm.py.utils.save_volfile(moved[t].squeeze(), save_path, affine)

        flow_filename = os.path.basename(fixed_path)[:-7] + '.npz'
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

newpath_flow = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Flow')
delete_if_exist(newpath_flow)
os.makedirs(newpath_flow)

newpath_registered = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Registered')
delete_if_exist(newpath_registered)
os.makedirs(newpath_registered)

newpath_seg = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Segmentation')
delete_if_exist(newpath_seg)
os.makedirs(newpath_seg)

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

add_feat_axis = not args.multichannel

config = read_config_video(os.path.join(os.path.dirname(args.model), 'config.yaml'))

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
    with open(os.path.join('splits', 'Lib', 'splits_final.pkl'), 'rb') as f:
        data = pickle.load(f)
        training_patients = data[0]['train']
        validation_patients = data[0]['val']
    if args.test_or_val == 'test':
        path_list = glob(os.path.join('voxelmorph_Lib_2D_testing', '*.gz'))
        path_list_gt = glob(os.path.join('voxelmorph_Lib_gt_2D_testing', '*.gz'))
        path_list_pkl = glob(os.path.join('voxelmorph_Lib_2D_testing', '*.pkl'))
    elif args.test_or_val == 'val':
        path_list = glob(os.path.join('voxelmorph_Lib_2D', '*.gz'))
        path_list_gt = glob(os.path.join('voxelmorph_Lib_2D_gt', '*.gz'))
        path_list_pkl = glob(os.path.join('voxelmorph_Lib_2D', '*.pkl'))
    image_size = 192

# load and set up model
# load the model

if 'motion_from_ed' not in config:
    config['motion_from_ed'] = False

network_1 = build_flow_model_successive(config, image_size=image_size, log_function=None, nb_channels=1)
if config['video_length'] > 2:
    nb_channels = 6
    network_2 = build_flow_model_successive(config, image_size=image_size, log_function=None, nb_channels=nb_channels)
else:
    network_2 = None
model = ModelWrap(model1=network_1, model2=network_2, do_ds=config['deep_supervision'], modality=config['training_modality'], motion_from_ed=config['motion_from_ed'])

model.load_state_dict(torch.load(args.model)['state_dict'])
model = model.cuda()
model.eval()

training_patients = sorted(list(set([x.split('_')[0] for x in training_patients])))
validation_patients = sorted(list(set([x.split('_')[0] for x in validation_patients])))

path_list = sorted([x for x in path_list if os.path.basename(x).split('_')[0] in validation_patients])
path_list_gt = sorted([x for x in path_list_gt if os.path.basename(x).split('_')[0] in validation_patients])
path_list_pkl = sorted([x for x in path_list_pkl if os.path.basename(x).split('_')[0] in validation_patients])

assert len(path_list) == len(path_list_pkl)

patient_list = sorted(list(set([os.path.basename(x).split('_')[0] for x in path_list])))

all_patient_paths = []
all_patient_paths_gt = []
all_patient_paths_pkl = []
for patient in patient_list:
    patient_files = []
    patient_files_pkl = []
    patient_files_gt = []
    for (path, pkl_path) in zip(path_list, path_list_pkl):
        if patient in path:
            patient_files.append(path)
        if patient in pkl_path:
            patient_files_pkl.append(pkl_path)
    for gt_path in path_list_gt:
        if patient in gt_path:
            patient_files_gt.append(gt_path)
    all_patient_paths.append(sorted(patient_files))
    all_patient_paths_pkl.append(sorted(patient_files_pkl))
    all_patient_paths_gt.append(sorted(patient_files_gt))


for (path_list_gz, path_list_pkl, path_list_gt) in tqdm(zip(all_patient_paths, all_patient_paths_pkl, all_patient_paths_gt), total=len(all_patient_paths)):

    path_list_gt_indices = [int(os.path.basename(x).split('frame')[-1][:2]) - 1 for x in path_list_gt]
    new_path_list_gt = [None] * len(path_list_gz)
    for i, gt_name in zip(path_list_gt_indices, path_list_gt):
        new_path_list_gt[i] = gt_name

    assert len(path_list_gz) == len(path_list_pkl) == len(new_path_list_gt)

    with open(path_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        ed_number = np.rint(data['ed_number']).astype(int)
        es_number = np.rint(data['es_number']).astype(int)

    path_list_gz = np.array(path_list_gz)
    new_path_list_gt = np.array(new_path_list_gt)
    frame_indices = np.arange(len(path_list_gz))
    after = frame_indices >= ed_number
    before = frame_indices < ed_number
    path_list_gz = np.concatenate([path_list_gz[after], path_list_gz[before]])
    new_path_list_gt = np.concatenate([new_path_list_gt[after], new_path_list_gt[before]])

    assert int(os.path.basename(path_list_gz[0]).split('frame')[-1][:2]) == ed_number + 1

    with torch.no_grad():
        if config['video_length'] > 2:
            inference_window_all_chunk_whole(path_list_gz, new_path_list_gt, newpath_flow, newpath_registered, newpath_seg, model, image_size, es_number, deep_supervision=config['deep_supervision'])
            #inference_window_all_chunk(path_list_gz, new_path_list_gt, newpath_flow, newpath_registered, newpath_seg, model, image_size, es_number, deep_supervision=config['deep_supervision'])
        else:
            if config['dataloader_modality'] == 'all_adjacent':
                inference_iterative_warp(path_list_gz, new_path_list_gt, newpath_flow, newpath_registered, model, image_size)
            else:
                inference_iterative(path_list_gz, new_path_list_gt, newpath_flow, newpath_registered, model, image_size)

    
