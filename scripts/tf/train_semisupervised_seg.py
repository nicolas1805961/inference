#!/usr/bin/env python

"""
Example script to train a VoxelMorph model in a semi-supervised
fashion by providing ground-truth segmentation data for training images.

If you use this code, please cite the following
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

from datetime import datetime
import pickle
import os
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from time import strftime
from ruamel.yaml import YAML
from glob import glob

def read_config_video(filename):
    yaml = YAML()
    with open(filename) as file:
        config = yaml.load(file)

    return config


# disable eager execution
#tf.compat.v1.disable_eager_execution()

# parse the commandline
parser = argparse.ArgumentParser()


# data organization parameters
#parser.add_argument('--img-list-moving', required=True, help='line-seperated list of training files')
#parser.add_argument('--img-list-fixed', required=True, help='line-seperated list of training files')
#parser.add_argument('--seg-list_moving', required=True, help='line-seperated list of training files')
#parser.add_argument('--seg-list_fixed', required=True, help='line-seperated list of training files')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', required=True, help='label list (npy format) to use in dice loss')
parser.add_argument('--config', required=True, help='config file')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--atlas', help='optional atlas to perform scan-to-atlas training')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=250,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--grad-loss-weight', type=float, default=0.02,
                    help='weight of gradient loss (lamba) (default: 0.02)(ncc=1.0)')
parser.add_argument('--dice-loss-weight', type=float, default=0.01,
                    help='weight of dice loss (gamma) (default: 0.01)(ncc=0.1)')
args = parser.parse_args()

config = read_config_video(args.config)

args.int_steps = config['int_steps']
args.image_loss = config['image_loss']
args.grad_loss_weight = config['grad_loss_weight']
args.dice_loss_weight = config['dice_loss_weight']
args.epochs = config['epochs']

all_paths_ed = glob(os.path.join(r"ED_ES_data", 'ED', '*.npy'))
all_paths_es = glob(os.path.join(r"ED_ES_data", 'ES', '*.npy'))

#train_imgs_moving = vxm.py.utils.read_file_list(args.img_list_moving)
#train_imgs_fixed = vxm.py.utils.read_file_list(args.img_list_fixed)
#train_segs_moving = vxm.py.utils.read_file_list(args.seg_list_moving)
#train_segs_fixed = vxm.py.utils.read_file_list(args.seg_list_fixed)
assert len(all_paths_ed) > 0, 'Could not find any training data.'
assert len(all_paths_es) > 0, 'Could not find any training data.'

# load labels file
train_labels = np.load(args.labels)

with open(os.path.join('splits', 'Lib', 'val', 'splits_final.pkl'), 'rb') as f:
    data = pickle.load(f)[0]

assert len(all_paths_ed) == len(all_paths_es)

all_paths_ed = sorted(all_paths_ed)
all_paths_es = sorted(all_paths_es)

train_paths_ed = []
val_paths_ed = []
for path_ed in all_paths_ed:
    payload = os.path.basename(path_ed).split('.')[0]
    if payload in data['train']:
        train_paths_ed.append(path_ed)
    elif payload in data['val']:
        val_paths_ed.append(path_ed)

train_paths_es = []
val_paths_es = []
for path_es in all_paths_es:
    payload = os.path.basename(path_es).split('.')[0]
    if payload in data['train']:
        train_paths_es.append(path_es)
    elif payload in data['val']:
        val_paths_es.append(path_es)

assert len(train_paths_ed) == len(train_paths_es)

# generator (scan-to-scan unless the atlas cmd argument was provided)
#generator = vxm.generators.semisupervised(
#    train_imgs_moving, train_imgs_fixed, train_segs_moving, train_segs_fixed, labels=train_labels, atlas_file=args.atlas)
generator = vxm.generators.semisupervised2(
    train_paths_ed, train_paths_es, labels=train_labels, atlas_file=args.atlas)

#generator = vxm.generators.semisupervised3D(
#    train_imgs, train_segs, labels=train_labels, atlas_file=args.atlas)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# unet architecture
#enc_nf = args.enc if args.enc else [16, 32, 32, 32]
#dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
enc_nf = args.enc if args.enc else [64, 192, 384, 768]
dec_nf = args.dec if args.dec else [768, 768, 384, 192, 96, 48, 16]

# prepare model checkpoint save path
timestr = datetime.now().strftime("%Y-%m-%d_%HH%M_%Ss_%f")
log_dir = os.path.join(model_dir, timestr)
save_filename = os.path.join(log_dir, '{epoch:04d}.h5')

# build the model
model = vxm.networks.VxmDenseSemiSupervisedSeg(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    nb_labels=len(train_labels),
    int_steps=args.int_steps,
    int_resolution=args.int_downsize,
    nb_unet_conv_per_level=1
)

# load initial weights (if provided)
if args.load_weights:
    model.load_weights(args.load_weights)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

ce = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.0,
    axis=-1,
    reduction="auto",
    name="categorical_crossentropy",
)

# losses
losses = [image_loss_func, vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss, vxm.losses.Dice().loss]
#weights = [0.0, args.grad_loss_weight, args.dice_loss_weight]
weights = [1.0, args.grad_loss_weight, args.dice_loss_weight]

#losses = [image_loss_func, vxm.losses.Dice().loss]
#weights = [1, args.dice_loss_weight]

# multi-gpu support
nb_devices = len(args.gpu.split(','))
if nb_devices > 1:
    save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
    model = tf.keras.utils.multi_gpu_model(model, gpus=nb_devices)
else:
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=10)

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr, epsilon=1e-3), loss=losses, loss_weights=weights)
#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=losses, loss_weights=weights)

# save starting weights
model.save(save_filename.format(epoch=args.initial_epoch))

yaml = YAML()
with open(os.path.join(log_dir, 'config.yaml'), 'wb') as f:
    yaml.dump(config, f)

with open(os.path.join(log_dir, 'log.txt'), 'w') as f:
    f.write(f'int_steps: {args.int_steps}\n')
    f.write(f'image_loss: {args.image_loss}\n')
    f.write(f'grad_loss_weight: {args.grad_loss_weight}\n')
    f.write(f'dice_loss_weight: {args.dice_loss_weight}\n')

model.fit_generator(generator,
                    initial_epoch=args.initial_epoch,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    callbacks=[save_callback],
                    verbose=1,
                    )
