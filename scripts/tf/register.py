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

import os
import argparse
from tqdm import tqdm
import pickle
from glob import glob
import voxelmorph as vxm
import tensorflow as tf
from pathlib import Path
import shutil

# third party
import numpy as np

def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

def split_seg_2d(seg):
    prob_seg = np.zeros((*seg.shape[:3], 4))
    for i, label in enumerate([0, 1, 2, 3]):
        prob_seg[0, ..., i] = seg[0, ..., 0] == label
    return prob_seg


def inference_iterative(path_list_gz, newpath_flow, newpath_registered, model, transform_object):
    fixed_path = path_list_gz[0]
    fixed, _ = vxm.py.utils.load_volfile(fixed_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    for t in range(len(path_list_gz)):
        moving_path = path_list_gz[t]

        if fixed_path == moving_path:
            continue

        filename = os.path.basename(moving_path)

        moving, moving_seg = vxm.py.utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=add_feat_axis)

        # load moving and fixed images

        out_list = []
        out_list_flow = []
        out_list_img = []
        for i in range(moving.shape[3]):
            current_moving = moving[:, :, :, i, :]
            current_moving_seg = moving_seg[:, :, :, i, :]
            current_fixed = fixed[:, :, :, i, :]

            current_moving_seg = split_seg_2d(current_moving_seg)

            #inshape = current_moving.shape[1:-1]
            #nb_feats = current_moving_seg.shape[-1]

            warp = model.register(current_moving, current_fixed)

            #fig, ax = plt.subplots(1, 1)
            #ax.imshow(warp[0, :, :, 0], cmap='plasma')
            #plt.show()

            moved = transform_object.predict([current_moving_seg, warp])
            #moved = np.argmax(moved, axis=-1, keepdims=True).astype(np.uint8)

            out_list.append(moved[0, :, :, :])
            out_list_flow.append(warp[0])
            out_list_img.append(current_moving[0, :, :, 0])
        
        moved = np.stack(out_list, axis=2) # H, W, D, C
        flow = np.stack(out_list_flow, axis=2)
        img = np.stack(out_list_img, axis=2)
        moved = moved.transpose(3, 0, 1, 2)

        # save moved image
        filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path = os.path.join(newpath_registered, filename)
        np.savez(save_path, seg=moved.squeeze())

        flow_filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow, img=img)


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

newpath_flow = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Backward_flow')
delete_if_exist(newpath_flow)
os.makedirs(newpath_flow)

newpath_registered = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Registered_backward')
delete_if_exist(newpath_registered)
os.makedirs(newpath_registered)

#log_dir = os.path.dirname(args.model)
# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

add_feat_axis = not args.multichannel

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

config = dict(inshape=(image_size, image_size), input_model=None)
model = vxm.networks.VxmDenseSemiSupervisedSeg.load(args.model, **config)
transform_object = vxm.networks.Transform((image_size, image_size), nb_feats=4)

validation_patients = sorted(list(set([x.split('_')[0] for x in validation_patients])))

path_list = sorted([x for x in path_list if os.path.basename(x).split('_')[0] in validation_patients])
path_list_pkl = sorted([x for x in path_list_pkl if os.path.basename(os.path.dirname(x)) in validation_patients])

assert len(path_list) == len(path_list_pkl)

patient_list = sorted(list(set([os.path.basename(x).split('_')[0] for x in path_list])))

all_patient_paths = []
all_patient_paths_pkl = []
for patient in tqdm(patient_list):
    patient_files = []
    patient_files_pkl = []
    for (path, pkl_path) in zip(path_list, path_list_pkl):
        if patient in path:
            patient_files.append(path)
        if patient in pkl_path:
            patient_files_pkl.append(pkl_path)
    all_patient_paths.append(sorted(patient_files))
    all_patient_paths_pkl.append(sorted(patient_files_pkl))

with tf.device(device):

    for (path_list_gz, path_list_pkl) in tqdm(zip(all_patient_paths, all_patient_paths_pkl), total=len(all_patient_paths)):

        assert len(path_list_gz) == len(path_list_pkl)

        with open(path_list_pkl[0], 'rb') as f:
            data = pickle.load(f)
            ed_number = np.rint(data['ed_number']).astype(int)

        path_list_gz = np.array(path_list_gz)
        frame_indices = np.arange(len(path_list_gz))
        after = frame_indices >= ed_number
        before = frame_indices < ed_number
        path_list_gz = np.concatenate([path_list_gz[after], path_list_gz[before]])

        patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

        current_newpath_flow = os.path.join(newpath_flow, patient_name)
        current_newpath_registered = os.path.join(newpath_registered, patient_name)

        delete_if_exist(current_newpath_flow)
        os.makedirs(current_newpath_flow)

        delete_if_exist(current_newpath_registered)
        os.makedirs(current_newpath_registered)

        assert int(os.path.basename(path_list_gz[0]).split('frame')[-1][:2]) == ed_number + 1

        inference_iterative(path_list_gz, current_newpath_flow, current_newpath_registered, model, transform_object)

    
