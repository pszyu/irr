#!/bin/bash
#SBATCH -t 200:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=p100
#SBATCH -o output/slurm-%j.out

# experiments and datasets meta
EXPERIMENTS_HOME="experiments"

# datasets
FLYINGTHINGS_HOME=/usr/xtmp/vision/datasets/FlyingThings3D_subset_filtered/
SINTEL_HOME=/usr/xtmp/vision/datasets/MPI_Sintel/

# model and checkpoint
MODEL=IRR_PWC_FEE_OEE
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample
CHECKPOINT="saved_check_point/IRR-PWC_flyingchairsOcc/checkpoint_latest.ckpt"
SIZE_OF_BATCH=3

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-$TIME"

# training configuration
python3 ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[128, 139, 149]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-5 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--start_epoch=109 \
--total_epochs=159 \
--training_augmentation=RandomAffineFlowOcc \
--training_augmentation_crop="[384,768]" \
--training_dataset=FlyingThings3dCleanTrain \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGTHINGS_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=F1 \
--validation_loss=$EVAL_LOSS \
--validation_key_minimize=False \
--fair_weights=True