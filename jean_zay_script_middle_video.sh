#!/bin/bash
#SBATCH --array=1-3   # Number of configuration files
#SBATCH --job-name=gpu_mono          # nom du job
#SBATCH -C v100-16g                  # reserver des GPU 16 Go seulement
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --qos=qos_gpu-t3            # qos_gpu-t4 qos_gpu-dev qos_gpu-t3
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference Ã  l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00          # 48:00:00 temps maximum d'execution demande (HH:MM:SS) 00:05:00 20:00:00  
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.11.0
#module load pytorch-gpu/py3/1.11.0

# echo des commandes lancees
set -x
#.\scripts\tf\register.py --test_or_val test --dataset Lib --dirpath results\VM-DIF-2 --model results\VM-DIF-2\0180.h5 -g 0
#python ./scripts/tf/register_2D.py --moving val_list_moving_2D.txt --fixed val_list_fixed_2D.txt --dirpath results --seg val_list_moving_seg_2D.txt --model 2023-09-24_23H18\0600.h5 --gpu 0
#python ./scripts/tf/register.py --moving val_list_moving.txt --fixed val_list_fixed.txt --dirpath results --seg val_list_moving_seg.txt --model results/0000.h5 --gpu 0
#python ./scripts/tf/train_semisupervised_seg.py --img-list train_list.txt --seg-list train_list_seg.txt --model-dir results --gpu 0 --labels labels.npy
#python ./scripts/tf/train_semisupervised_seg.py --img-list-moving train_list_moving_2D_Lib.txt --img-list-fixed train_list_fixed_2D_Lib.txt --seg-list_moving train_list_moving_seg_2D_Lib.txt --seg-list_fixed train_list_fixed_seg_2D_Lib.txt --model-dir results --gpu 0 --labels labels.npy
#python ./scripts/torch/train.py --img-list-moving train_list_moving_2D_Lib.txt --img-list-fixed train_list_fixed_2D_Lib.txt --seg-list_moving train_list_moving_seg_2D_Lib.txt --seg-list_fixed train_list_fixed_seg_2D_Lib.txt --model-dir results --gpu 0 --labels labels.npy
#python ./scripts/tf/train_semisupervised_seg.py --img-list-moving train_list_moving_2D_Lib.txt --img-list-fixed train_list_fixed_2D_Lib.txt --seg-list_moving train_list_moving_seg_2D_Lib.txt --seg-list_fixed train_list_fixed_seg_2D_Lib.txt --model-dir results --gpu 0 --labels labels.npy --config config${SLURM_ARRAY_TASK_ID}.yaml
python ./scripts/tf/train_semisupervised_seg.py --model-dir results --gpu 0 --labels labels.npy --config config${SLURM_ARRAY_TASK_ID}.yaml

sleep 10