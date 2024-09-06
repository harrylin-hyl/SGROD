#!/usr/bin/env bash

echo running training of SGROD, S-OWODB dataset

set -x

#####------------------------------------------------------SAM---------------------------------------------------------------#####
EXP_DIR=exps/SOWODB/SGROD
PY_ARGS=${@:1}
WANDB_NAME=S_SGROD

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19\
    --train_set 'owdetr_t1_train' --test_set 'owdetr_test' --epochs 41 --lr_drop 31\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owdetr_t1_ft.txt"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}

PY_ARGS=${@:1}    
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" --dataset OWDETR --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21\
    --train_set 'owdetr_t2_train' --test_set 'owdetr_test' --epochs 51\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 --freeze_prob_model\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t2"\
    --exemplar_replay_selection --exemplar_replay_max_length 1679 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owdetr_t1_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t2_ft.txt"\
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --lr 2e-5\
    --iou_thre 0.4 --obj_thre 0.7  --sns_thre 0.3 --lr_decline_p 0.5  --unmatched_boxes \
    ${PY_ARGS}
    

# PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
    --train_set "${WANDB_NAME}/learned_owdetr_t2_ft" --test_set 'owdetr_test' --epochs 121 --lr_drop 50\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t2_ft"\
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes\
    ${PY_ARGS}


# PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" --dataset OWDETR --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20\
    --train_set 'owdetr_t3_train' --test_set 'owdetr_test' --epochs 131\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 --freeze_prob_model\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t3"\
    --exemplar_replay_selection --exemplar_replay_max_length 2345 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owdetr_t2_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t3_ft.txt"\
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0120.pth" --lr 2e-5\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}
    
    
# PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "${WANDB_NAME}/learned_owdetr_t3_ft" --test_set 'owdetr_test' --epochs 201 --lr_drop 50\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5\
    --wandb_name "${WANDB_NAME}_t3_ft"\
    --pretrain "${EXP_DIR}/t3/checkpoint0130.pth"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}


PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set 'owdetr_t4_train' --test_set 'owdetr_test' --epochs 211 \
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 --freeze_prob_model\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t4"\
    --exemplar_replay_selection --exemplar_replay_max_length 2664 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owdetr_t3_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t4_ft.txt"\
    --num_inst_per_class 40\
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0200.pth" --lr 2e-5\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}
    
    
# PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set "${WANDB_NAME}/learned_owdetr_t4_ft" --test_set 'owdetr_test' --epochs 301 --lr_drop 50\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t4_ft"\
    --pretrain "${EXP_DIR}/t4/checkpoint0210.pth"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}