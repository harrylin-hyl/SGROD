#!/usr/bin/env bash

echo running eval of SGROD, M-OWODB dataset

set -x

EXP_DIR=checkpoints/SGROD/MOWODB
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0\
    ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0\
    ${PY_ARGS}

PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 8e-4 --obj_temp 1.0\
    --pretrain "${EXP_DIR}/t4/_ft/checkpoint0260.pth" --eval --wandb_project ""\
    --post_geometirc_mean_alpha 1.0\
    ${PY_ARGS}
    
    

    
    
    
    