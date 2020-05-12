#!/bin/bash
#SBATCH -t 200:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=p100
#SBATCH -o output/slurm-%j.out

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
FLYINGCHAIRS_OCC_HOME=/usr/xtmp/shuzhiyu/datasets/FCOcc/FlyingChairsOcc/data

# model and checkpoint
MODEL=IRR_PWC_FED_OEE
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample
CHECKPOINT="experiments/IRR_PWC_FED_OEE-20200506-153207/checkpoint_latest.ckpt"
SIZE_OF_BATCH=3

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-2-fc2-$TIME"

# training configuration
python3 ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[54, 72, 90]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=66 \
--total_epochs=108 \
--training_augmentation=RandomAffineFlowOcc \
--training_dataset=FlyingChairsOccTrain \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGCHAIRS_OCC_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=FlyingChairsOccValid  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$FLYINGCHAIRS_OCC_HOME \
--validation_key=F1 \
--validation_loss=$EVAL_LOSS \
--validation_key_minimize=False \
--fair_weights=True