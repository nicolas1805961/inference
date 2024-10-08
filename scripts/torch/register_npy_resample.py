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
from nnunet.lib.training_utils import read_config_video, build_seg_flow_gaussian_model, read_config, build_2d_model
from nnunet.network_architecture.integration import SpatialTransformer
from nnunet.lib.utils import ConvBlocks2DGroup

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
import matplotlib
import kornia


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




def inference_window_all_chunk_whole(path_list_gz, newpath_flow_forward, newpath_flow_backward, newpath_registered_forward, newpath_registered_backward, model, image_size, es_number, label_input, binary_distance_input, distance_map_power):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

    newpath_flow_forward = os.path.join(newpath_flow_forward, patient_name)
    newpath_flow_backward = os.path.join(newpath_flow_backward, patient_name)
    newpath_registered_forward = os.path.join(newpath_registered_forward, patient_name)
    newpath_registered_backward = os.path.join(newpath_registered_backward, patient_name)

    delete_if_exist(newpath_flow_forward)
    os.makedirs(newpath_flow_forward)

    delete_if_exist(newpath_flow_backward)
    os.makedirs(newpath_flow_backward)

    delete_if_exist(newpath_registered_forward)
    os.makedirs(newpath_registered_forward)

    delete_if_exist(newpath_registered_backward)
    os.makedirs(newpath_registered_backward)

    es_idx = np.where([int(os.path.basename(path_list_gz[i]).split('frame')[-1][:2]) == es_number + 1 for i in range(len(path_list_gz))])[0][0]

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

    out_list_backward = []
    out_list_flow_backward = []
    out_list_img_backward = []
    for d in range(x.shape[-1]):
        current_x = x[:, :, :, :, :, d]
        current_mask = mask[:, :, :, :, :, d]

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(5, 4)
        #for k in range(4):
        #    ax[0, k].imshow(current_x[k, 0, 0].cpu(), cmap='gray')
        #    ax[1, k].imshow(gt[k, 0, 0, :, :, d].cpu(), cmap='gray')
        #    ax[2, k].imshow(current_mask[k, 0, 0].cpu(), cmap='hot')
        #    ax[3, k].imshow(current_mask[k, 0, 1].cpu(), cmap='hot')
        #    ax[4, k].imshow(current_mask[k, 0, 2].cpu(), cmap='hot')
        #plt.show()

        indices = torch.arange(1, len(current_x))
        chunk1 = indices[:es_idx-1]
        chunk2 = indices[es_idx-1:]
        chunk2 = torch.flip(chunk2, dims=[0])

        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])

        #if pretrained_network:
        #    with torch.no_grad():
        #        first_input = NormalizeIntensity()(torch.clone(current_x[0]))
        #        initial_mask = pretrained_network(first_input)['pred']
        #        #initial_mask = torch.softmax(initial_mask, dim=1)
#
        #        #fig, ax = plt.subplots(1, 3)
        #        #ax[0].imshow(current_x[0, 0, 0].cpu(), cmap='gray')
        #        #ax[1].imshow(initial_mask.cpu()[0, 2], cmap='plasma')
        #        #ax[2].imshow(gt[0, 0, 0, :, :, d].cpu(), cmap='gray')
        #        #plt.show()
        #else:
        #    initial_mask = gt[0, :, :, :, :, d]
        #    initial_mask = torch.nn.functional.one_hot(initial_mask[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()

        chunk_list_backward_flow = []

        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
            
            chunk_x = current_x[chunk]
            chunk_mask = current_mask[chunk]

            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(chunk_x[0, 0, 0].cpu(), cmap='gray')
            #ax[1].imshow(chunk_mask.cpu()[0, 0, 0].float(), cmap='hot', vmin=0.0, vmax=1.0)
            #ax[2].imshow(chunk_mask.cpu()[0, 0, 1].float(), cmap='hot', vmin=0.0, vmax=1.0)
            #ax[3].imshow(chunk_mask.cpu()[0, 0, 2].float(), cmap='hot', vmin=0.0, vmax=1.0)
            #plt.show()
            #exit(0)

            chunk_x = NormalizeIntensity()(chunk_x)

            with torch.no_grad():
                if label_input:
                    chunk_mask = torch.pow(4 * torch.exp(-chunk_mask) / ((1 + torch.exp(-chunk_mask))**2), distance_map_power)

                    #fig, ax = plt.subplots(1, 3)
                    #ax[0].imshow(chunk_mask[0, 0, 0].cpu(), cmap='hot')
                    #ax[1].imshow(chunk_mask[0, 0, 1].cpu(), cmap='hot')
                    #ax[2].imshow(chunk_mask[0, 0, 2].cpu(), cmap='hot')
                    #plt.show()
                    
                    if binary_distance_input:
                        chunk_mask = chunk_mask.long().float()
                    out = model(chunk_x, label=chunk_mask, step=1)
                else:
                    out = model(chunk_x, step=1)
            backward_flow_pred = out['backward_flow'].detach().to('cpu')
            #del chunk_x
            torch.cuda.empty_cache()

            chunk_list_backward_flow.append(backward_flow_pred)

        nan_tensor_1 = torch.full(size=(len(chunk1),) + chunk_list_backward_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_backward_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_backward_flow[0].to('cuda:0')
        nan_tensor_2 = chunk_list_backward_flow[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        warp = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        assert len(warp) == len(current_x) - 1

        registered_list = []
        for t in range(len(warp)):
            moving_seg = gt[t + 1]
            current_moving_seg = moving_seg[:, :, :, :, d]
            ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()

            registered = motion_estimation(flow=warp[t], original=ed_target, mode='bilinear')
            #moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W
            registered_list.append(registered)
        moved_backward = torch.stack(registered_list, dim=0)
        assert len(moved_backward) == len(current_x) - 1


        out_list_backward.append(moved_backward[:, 0])
        out_list_flow_backward.append(warp[:, 0].permute(0, 2, 3, 1).contiguous())
        out_list_img_backward.append(current_x[1:, 0, 0])
    
    moved_backward = torch.stack(out_list_backward, dim=4).detach().cpu().numpy() # T - 1, C, H, W, D
    flow_backward = torch.stack(out_list_flow_backward, dim=3).detach().cpu().numpy() # T - 1, H, W, D, 2
    img_backward = torch.stack(out_list_img_backward, dim=3).detach().cpu().numpy() # T - 1, H, W, D

    for t in range(len(moved_backward)):
        # save moved image
        moving_path = path_list_gz[t+1]

        filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path = os.path.join(newpath_registered_backward, filename)
        np.savez(save_path, seg=moved_backward[t].squeeze())

        flow_filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path_flow = os.path.join(newpath_flow_backward, flow_filename)
        np.savez(save_path_flow, flow=flow_backward[t], img=img_backward[t])






#def inference_window_all_chunk_whole(path_list_gz, path_list_gt, newpath_flow_forward, newpath_flow_backward, newpath_registered_forward, newpath_registered_backward, model, image_size, es_number, pretrained_network):
#    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')
#
#    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]
#
#    newpath_flow_forward = os.path.join(newpath_flow_forward, patient_name)
#    newpath_flow_backward = os.path.join(newpath_flow_backward, patient_name)
#    newpath_registered_forward = os.path.join(newpath_registered_forward, patient_name)
#    newpath_registered_backward = os.path.join(newpath_registered_backward, patient_name)
#
#    delete_if_exist(newpath_flow_forward)
#    os.makedirs(newpath_flow_forward)
#
#    delete_if_exist(newpath_flow_backward)
#    os.makedirs(newpath_flow_backward)
#
#    delete_if_exist(newpath_registered_forward)
#    os.makedirs(newpath_registered_forward)
#
#    delete_if_exist(newpath_registered_backward)
#    os.makedirs(newpath_registered_backward)
#
#    es_idx = np.where([int(os.path.basename(path_list_gz[i]).split('frame')[-1][:2]) == es_number + 1 for i in range(len(path_list_gz))])[0][0]
#
#    x_list = []
#    gt_list = []
#    for (path_gz, path_gt) in zip(path_list_gz, path_list_gt):
#        x, fixed_affine = vxm.py.utils.load_volfile(path_gz, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
#        x = torch.from_numpy(x).to(device).float().permute(0, 4, 1, 2, 3)
#
#        gt = vxm.py.utils.load_volfile(path_gt, add_batch_axis=True, add_feat_axis=add_feat_axis)
#        gt = torch.from_numpy(gt).to(device).float().permute(0, 4, 1, 2, 3)
#        
#        x_list.append(x)
#        gt_list.append(gt)
#    
#    x = torch.stack(x_list, dim=0)
#    gt = torch.stack(gt_list, dim=0)
#
#    out_list_forward = []
#    out_list_backward = []
#    out_list_flow_forward = []
#    out_list_flow_backward = []
#    out_list_img_backward = []
#    out_list_img_forward = []
#    for d in range(x.shape[-1]):
#        current_x = x[:, :, :, :, :, d]
#
#        #indices = np.arange(len(current_x))
#        #indices1 = indices[:es_idx + 1]
#        #indices2 = indices[es_idx:]
#        #indices2 = np.concatenate([np.array(indices[0]).reshape(1,), indices2[::-1]])
#        #print(indices1)
#        #print(indices2)
#
#        indices = torch.arange(1, len(current_x))
#        chunk1 = indices[:es_idx-1]
#        chunk2 = indices[es_idx-1:]
#        chunk2 = torch.flip(chunk2, dims=[0])
#
#        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
#        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])
#
#        if pretrained_network:
#            with torch.no_grad():
#                first_input = NormalizeIntensity()(torch.clone(current_x[0]))
#                initial_mask = pretrained_network(first_input)['pred']
#                #initial_mask = torch.softmax(initial_mask, dim=1)
#
#                #fig, ax = plt.subplots(1, 3)
#                #ax[0].imshow(current_x[0, 0, 0].cpu(), cmap='gray')
#                #ax[1].imshow(initial_mask.cpu()[0, 2], cmap='plasma')
#                #ax[2].imshow(gt[0, 0, 0, :, :, d].cpu(), cmap='gray')
#                #plt.show()
#        else:
#            initial_mask = gt[0, :, :, :, :, d]
#            initial_mask = torch.nn.functional.one_hot(initial_mask[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
#
#        chunk_list_forward_flow = []
#        chunk_list_backward_flow = []
#
#        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
#            
#            chunk_x = current_x[chunk]
#
#            #fig, ax = plt.subplots(1, 2)
#            #ax[0].imshow(chunk_x[0, 0, 0].cpu(), cmap='gray')
#            #ax[1].imshow(initial_mask.cpu()[0, 3], cmap='plasma')
#            #plt.show()
#
#            chunk_x = NormalizeIntensity()(chunk_x)
#
#            with torch.no_grad():
#                #out = model(chunk_x, None, step=1)
#                out = model(chunk_x, initial_mask, step=1)
#            forward_flow_pred = out['forward_flow'].detach().to('cpu')
#            backward_flow_pred = out['backward_flow'].detach().to('cpu')
#            #del chunk_x
#            torch.cuda.empty_cache()
#
#            plot_coords_deformable(out['sampling_locations'], out['attention_weights'], images=chunk_x)
#
#            #step = len(chunk_x) / config['video_length']
#            #right_frames = []
#            #for t in range(config['video_length']):
#            #    index = int(t * step)
#            #    right_frames.append(chunk_x[index])
#            #right_frames = torch.stack(right_frames, dim=0)
#            #plot_coords_deformable(out['sampling_locations'], out['attention_weights'], images=right_frames)
#            
#            #fig, ax = plt.subplots(1, 2)
#            #ax[0].imshow(torch.argmax(seg_pred, dim=2)[5, 0].cpu(), cmap='gray')
#            #ax[1].imshow(initial_mask.cpu()[0, 2], cmap='plasma')
#            #plt.show()
#
#            chunk_list_forward_flow.append(forward_flow_pred)
#            chunk_list_backward_flow.append(backward_flow_pred)
#
#        nan_tensor_1 = torch.full(size=(len(chunk1),) + chunk_list_forward_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
#        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_forward_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
#        nan_tensor_1 = chunk_list_forward_flow[0].to('cuda:0')
#        nan_tensor_2 = chunk_list_forward_flow[1].to('cuda:0')
#
#        assert torch.all(torch.isfinite(nan_tensor_1))
#        assert torch.all(torch.isfinite(nan_tensor_2))
#        
#        warp_forward = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)
#
#        assert len(warp_forward) == len(current_x) - 1
#
#        nan_tensor_1 = torch.full(size=(len(chunk1) + 1,) + chunk_list_backward_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
#        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_backward_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
#        nan_tensor_1 = chunk_list_backward_flow[0].to('cuda:0')
#        nan_tensor_2 = chunk_list_backward_flow[1].to('cuda:0')
#
#        assert torch.all(torch.isfinite(nan_tensor_1))
#        assert torch.all(torch.isfinite(nan_tensor_2))
#        
#        warp_backward = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)
#
#
#        moving_seg = gt[0]
#        current_moving_seg = moving_seg[:, :, :, :, d]
#        ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
#        
#        registered_list_forward = [ed_target]
#        for t in range(len(warp_forward)):
#
#            registered = motion_estimation(flow=warp_forward[t], original=ed_target, mode='bilinear')
#            #moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W
#            registered_list_forward.append(registered)
#        moved_forward = torch.stack(registered_list_forward, dim=0)
#        assert len(moved_forward) == len(current_x)
#
#
#        registered_list = []
#        for t in range(len(warp_backward)):
#            moving_seg = gt[t + 1]
#            current_moving_seg = moving_seg[:, :, :, :, d]
#            ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
#
#            registered = motion_estimation(flow=warp_backward[t], original=ed_target, mode='bilinear')
#            #moved = torch.argmax(registered, dim=1, keepdim=True).int() # B, 1, H, W
#            registered_list.append(registered)
#        moved_backward = torch.stack(registered_list, dim=0)
#        assert len(moved_backward) == len(current_x) - 1
#
#
#        out_list_forward.append(moved_forward[:, 0])
#        out_list_backward.append(moved_backward[:, 0])
#        out_list_flow_forward.append(warp_forward[:, 0].permute(0, 2, 3, 1).contiguous())
#        out_list_flow_backward.append(warp_backward[:, 0].permute(0, 2, 3, 1).contiguous())
#        out_list_img_backward.append(current_x[1:, 0, 0])
#        out_list_img_forward.append(current_x[0, 0, 0][None].repeat(len(warp_forward), 1, 1))
#    
#    moved_forward = torch.stack(out_list_forward, dim=4).detach().cpu().numpy() # T, C, H, W, D
#    moved_backward = torch.stack(out_list_backward, dim=4).detach().cpu().numpy() # T - 1, C, H, W, D
#    flow_forward = torch.stack(out_list_flow_forward, dim=3).detach().cpu().numpy() # T - 1, H, W, D, 2
#    flow_backward = torch.stack(out_list_flow_backward, dim=3).detach().cpu().numpy() # T - 1, H, W, D, 2
#    img_forward = torch.stack(out_list_img_forward, dim=3).detach().cpu().numpy() # T - 1, H, W, D
#    img_backward = torch.stack(out_list_img_backward, dim=3).detach().cpu().numpy() # T - 1, H, W, D
#
#    filename = os.path.basename(path_list_gz[0])[:-7] + '.npz'
#    save_path = os.path.join(newpath_registered_forward, filename)
#    np.savez(save_path, seg=moved_forward[0].squeeze())
#
#    for t in range(1, len(moved_forward)):
#        # save moved image
#        moving_path = path_list_gz[t]
#
#        filename = os.path.basename(moving_path)[:-7] + '.npz'
#        save_path = os.path.join(newpath_registered_forward, filename)
#        np.savez(save_path, seg=moved_forward[t].squeeze())
#
#        filename = os.path.basename(moving_path)[:-7] + '.npz'
#        save_path = os.path.join(newpath_registered_backward, filename)
#        np.savez(save_path, seg=moved_backward[t - 1].squeeze())
#
#        flow_filename = os.path.basename(moving_path)[:-7] + '.npz'
#        save_path_flow = os.path.join(newpath_flow_forward, flow_filename)
#        np.savez(save_path_flow, flow=flow_forward[t - 1], img=img_forward[t - 1])
#
#        flow_filename = os.path.basename(moving_path)[:-7] + '.npz'
#        save_path_flow = os.path.join(newpath_flow_backward, flow_filename)
#        np.savez(save_path_flow, flow=flow_backward[t - 1], img=img_backward[t - 1])




def inference_iterative_warp(path_list_gz, newpath_flow, newpath_registered, model, image_size, es_number):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    es_idx = np.where([int(os.path.basename(path_list_gz[i]).split('frame')[-1][:2]) == es_number + 1 for i in range(len(path_list_gz))])[0][0]

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

    flow_list_all = []
    moved_list_all = []
    img_list_all = []
    for d in range(x.shape[-1]):
        current_data_depth_seg = gt[:, :, :, :, :, d]
        current_data_depth = x[:, :, :, :, :, d]

        indices = torch.arange(1, len(current_data_depth))
        chunk1 = indices[:es_idx-1]
        chunk2 = indices[es_idx-1:]
        chunk2 = torch.flip(chunk2, dims=[0])

        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])

        chunk_list_flow = []
        chunk_list_seg = []

        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
            chunk_x = current_data_depth[chunk]
            chunk_seg = current_data_depth_seg[chunk]

            out_list_flow_depth = []

            for t in range(len(chunk_x) - 1):
                x_in = torch.stack([chunk_x[t], chunk_x[t + 1]], dim=0)
                x_in = NormalizeIntensity()(x_in)

                out = model(chunk_x, step=1)
                warp = out['backward_flow'][0] # B, C, H, W

                out_list_flow_depth.append(warp)
        
            flow = torch.stack(out_list_flow_depth, dim=0) # T, B, C, H, W
            assert len(flow) == len(chunk_x) - 1

            moved_list = []
            for t1 in reversed(range(1, len(chunk_seg))):
                current_moving_seg = chunk_seg[t1]
                ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
                for t2 in reversed(range(t1)):
                    ed_target = motion_estimation(flow=flow[t2], original=ed_target, mode='bilinear')
                #moved = torch.argmax(ed_target, dim=1, keepdim=True).int() # B, 1, H, W
                moved = ed_target # B, 4, H, W
                moved_list.append(moved)
            moved = torch.stack(moved_list, dim=0) # T, B, 4, H, W
            assert len(moved) == len(chunk_x) - 1
            moved = moved.flip(0)

            chunk_list_flow.append(flow)
            chunk_list_seg.append(moved)
        
        nan_tensor_1 = torch.full(size=(len(chunk1),) + chunk_list_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_flow[0].to('cuda:0')
        nan_tensor_2 = chunk_list_flow[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        warp = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        assert len(warp) == len(current_data_depth) - 1

        nan_tensor_1 = torch.full(size=(len(chunk1) + 1,) + chunk_list_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_seg[0].to('cuda:0')
        nan_tensor_2 = chunk_list_seg[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        seg = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        #moved_list = []
        #for t in range(len(flow)):
        #    current_moving_seg = current_data_depth_seg[t + 1]
        #    ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        #    for t2 in reversed(range(t + 1)):
        #        ed_target = motion_estimation(flow=flow[t2], original=ed_target, mode='bilinear')
        #    moved = torch.argmax(ed_target, dim=1, keepdim=True).int() # B, 1, H, W
        #    moved_list.append(moved)
        #moved = torch.stack(moved_list, dim=0)
        #assert len(moved) == len(current_data_depth) - 1

        flow_list_all.append(warp.cpu())
        moved_list_all.append(seg.cpu())
        img_list_all.append(current_data_depth[1:].cpu())
    
    moved = np.stack(moved_list_all, axis=-1) # T-1, 1, 4, H, W, D
    img = np.stack(img_list_all, axis=-1) # T-1, 1, 1, H, W, D
    flow = np.stack(flow_list_all, axis=-1) # T-1, 1, 2, H, W, D
    flow = flow.transpose(0, 1, 3, 4, 5, 2) # T-1, 1, H, W, D, 2

    for t in range(len(moved)):
        # save moved image
        moving_path = path_list_gz[t + 1]

        filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path = os.path.join(newpath_registered, filename)
        np.savez(save_path, seg=moved[t].squeeze())

        flow_filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow[t].squeeze(), img=img[t].squeeze())





def inference_iterative_warp_sum(path_list_gz, newpath_flow, newpath_registered, model, image_size, es_number):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    es_idx = np.where([int(os.path.basename(path_list_gz[i]).split('frame')[-1][:2]) == es_number + 1 for i in range(len(path_list_gz))])[0][0]

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

    flow_list_all = []
    moved_list_all = []
    img_list_all = []
    for d in range(x.shape[-1]):
        current_data_depth_seg = gt[:, :, :, :, :, d]
        current_data_depth = x[:, :, :, :, :, d]

        indices = torch.arange(1, len(current_data_depth))
        chunk1 = indices[:es_idx-1]
        chunk2 = indices[es_idx-1:]
        chunk2 = torch.flip(chunk2, dims=[0])

        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])

        chunk_list_flow = []
        chunk_list_seg = []

        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
            chunk_x = current_data_depth[chunk]
            chunk_seg = current_data_depth_seg[chunk]

            out_list_flow_depth = []

            for t in range(len(chunk_x) - 1):
                x_in = torch.stack([chunk_x[t], chunk_x[t + 1]], dim=0)
                x_in = NormalizeIntensity()(x_in)

                out = model(chunk_x, step=1)
                warp = out['backward_flow'][0] # B, C, H, W

                out_list_flow_depth.append(warp)
        
            flow = torch.stack(out_list_flow_depth, dim=0) # T, B, C, H, W
            assert len(flow) == len(chunk_x) - 1

            out_list_flow_depth = []
            cumulative = flow[0]
            for t1 in range(len(flow)):
                next_flow = flow[t1]
                #fig, ax = plt.subplots(1, 1)
                ##arr = arr.transpose((1, 0, 2, 3))
                #step_quiver = 4
                #X, Y = np.meshgrid(np.arange(0, cumulative.shape[-2], step=step_quiver), np.arange(cumulative.shape[-1], step=step_quiver))
                #ax.imshow(img_sequence[0, 0, 0].cpu(), cmap='gray')
                #ax.quiver(X, Y, cumulative[0, 1, ::step_quiver, ::step_quiver].cpu(), cumulative[0, 0, ::step_quiver, ::step_quiver].cpu(), color='r', angles='xy', scale_units='xy', scale=1)
                #plt.show()
                next_flow_x = next_flow[:, 0][:, None]
                next_flow_y = next_flow[:, 1][:, None]
                next_flow_x = motion_estimation(flow=cumulative, original=next_flow_x, mode='bilinear')
                next_flow_y = motion_estimation(flow=cumulative, original=next_flow_y, mode='bilinear')
                next_flow = torch.cat([next_flow_x, next_flow_y], dim=1)
                cumulative = cumulative + next_flow
                out_list_flow_depth.append(cumulative)
            flow = torch.stack(out_list_flow_depth, dim=0) # T, B, C, H, W
            assert len(flow) == len(chunk_x) - 1

            moved_list = []
            for t in range(len(flow)):
                current_moving_seg = chunk_seg[t + 1]
                ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
                moved = motion_estimation(flow=flow[t], original=ed_target, mode='bilinear')
                #moved = torch.argmax(moved, dim=1, keepdim=True).int() # B, 1, H, W
                moved = moved # B, 4, H, W
                moved_list.append(moved)
            moved = torch.stack(moved_list, dim=0)
            assert len(moved) == len(chunk_x) - 1

            chunk_list_flow.append(flow)
            chunk_list_seg.append(moved)
        
        nan_tensor_1 = torch.full(size=(len(chunk1),) + chunk_list_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_flow[0].to('cuda:0')
        nan_tensor_2 = chunk_list_flow[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        warp = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        assert len(warp) == len(current_data_depth) - 1

        nan_tensor_1 = torch.full(size=(len(chunk1) + 1,) + chunk_list_flow[0].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_2 = torch.full(size=(len(chunk2),) + chunk_list_flow[1].shape[1:], fill_value=torch.nan, device='cuda:0')
        nan_tensor_1 = chunk_list_seg[0].to('cuda:0')
        nan_tensor_2 = chunk_list_seg[1].to('cuda:0')

        assert torch.all(torch.isfinite(nan_tensor_1))
        assert torch.all(torch.isfinite(nan_tensor_2))
        
        seg = torch.cat([nan_tensor_1, torch.flip(nan_tensor_2, dims=[0])], dim=0)

        #moved_list = []
        #for t in range(len(flow)):
        #    current_moving_seg = current_data_depth_seg[t + 1]
        #    ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        #    for t2 in reversed(range(t + 1)):
        #        ed_target = motion_estimation(flow=flow[t2], original=ed_target, mode='bilinear')
        #    moved = torch.argmax(ed_target, dim=1, keepdim=True).int() # B, 1, H, W
        #    moved_list.append(moved)
        #moved = torch.stack(moved_list, dim=0)
        #assert len(moved) == len(current_data_depth) - 1

        flow_list_all.append(warp.cpu())
        moved_list_all.append(seg.cpu())
        img_list_all.append(current_data_depth[1:].cpu())
    
    moved = np.stack(moved_list_all, axis=-1) # T-1, 1, 1, H, W, D
    img = np.stack(img_list_all, axis=-1) # T-1, 1, 1, H, W, D
    flow = np.stack(flow_list_all, axis=-1) # T-1, 1, 2, H, W, D
    flow = flow.transpose(0, 1, 3, 4, 5, 2) # T-1, 1, H, W, D, 2

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

model = build_seg_flow_gaussian_model(config, image_size=image_size, log_function=None)

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
        if config['video_length'] > 2:
            inference_window_all_chunk_whole(path_list_gz, newpath_flow_forward, newpath_flow_backward, newpath_registered_forward, newpath_registered_backward, model, image_size, es_number, label_input=config['label_input'], binary_distance_input=config['binary_distance_input'], distance_map_power=config['distance_map_power'])
        else:
            inference_iterative_warp(path_list_gz, newpath_flow_backward, newpath_registered_backward, model, image_size, es_number)
            #inference_iterative_warp_sum(path_list_gz, newpath_flow_backward, newpath_registered_backward, model, image_size, es_number)
    
