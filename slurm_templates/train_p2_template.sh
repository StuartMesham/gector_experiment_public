#!/bin/sh
#SBATCH -J {{JOB_NAME}}
#SBATCH -A {{ACCOUNT}}
#SBATCH --nodes={{N_NODES}}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{{N_GPUS}}
#SBATCH --time={{JOB_TIME}}
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH -p {{PARTITION}}
#SBATCH --output={{SLURM_OUTPUT_FILE}}
#SBATCH --error={{SLURM_ERROR_FILE}}

export MASTER_ADDR=`/bin/hostname -s`
export WANDB_PROJECT={{WANDB_PROJECT}}
export HF_HOME={{HF_HOME}}

source {{VENV}}

echo "starting on node $SLURM_PROCID"

find {{BASE_MODEL}} -type d -name "*checkpoint-*" | xargs rm -r

python -m torch.distributed.launch \
--nproc_per_node={{N_GPUS}} \
--nnodes=$SLURM_NTASKS \
--master_port={{MASTER_PORT}} \
--node_rank=$SLURM_PROCID \
run_gector.py \
--do_train True \
--do_eval True \
--do_predict False \
--model_name_or_path {{BASE_MODEL}} \
--is_gector_model {{IS_GECTOR_MODEL}} \
--train_file {{TRAIN_DATA}} \
--validation_file {{DEV_DATA}} \
--tagset_file {{TAGSET_FILE}} \
--label_column_name labels \
--lr_scheduler_type constant \
--output_dir {{OUTPUT_DIR}} \
--early_stopping_patience 3 \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--evaluation_strategy epoch \
--save_strategy epoch \
--save_total_limit 1 \
--num_train_epochs 15 \
--logging_steps 1000 \
--cold_epochs 2 \
--cold_lr 1e-3 \
--learning_rate 1e-5 \
--optim adamw_torch \
--classifier_dropout 0.0 \
--per_device_train_batch_size {{BATCH_SIZE}} \
--per_device_eval_batch_size {{BATCH_SIZE}} \
--weight_decay 0 \
--report_to wandb_resume \
--wandb_run_id {{WANDB_RUN_ID}} \
--seed {{SEED}} \
--dataloader_num_workers 1
