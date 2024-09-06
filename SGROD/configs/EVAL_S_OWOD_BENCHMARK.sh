#!/usr/bin/env bash

echo running eval of SGROD, S-OWODB dataset

set -x

EXP_DIR=checkpoints/SGROD/SOWODB
PY_ARGS=${@:1}
 

PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
    --train_set "owdetr_t1_train" --test_set 'owdetr_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0 \
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset OWDETR --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
    --train_set "owdetr_t2_train" --test_set 'owdetr_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0120.pth"  --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0 \
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset OWDETR --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "owdetr_t3_train" --test_set 'owdetr_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0200.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0 \
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set "owdetr_t4_train" --test_set 'owdetr_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t4_ft/checkpoint0300.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0 \
    ${PY_ARGS}
    
    