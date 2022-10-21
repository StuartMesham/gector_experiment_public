#!/bin/bash

model=$1
tagset_size=$2

seeds=(0 42 52 62 72 82 92)  # 0 to make indices start at 1

# get shortened model name
model_name=$(basename $model)
if [[ $model_name == *-cased ]]
then
  model_name="${model_name: :-6}"
fi
if [[ $model_name == *-discriminator ]]
then
  model_name="${model_name: :-14}"
fi


# generate tagset_size_literal
if [[ $tagset_size == 5k ]]
then
  tagset_size_literal=5001
elif [[ $tagset_size == 10k ]]
then
  tagset_size_literal=10001
else
  echo invalid tagset size
  exit 1
fi


for run in 1 2 3 4 5 6
do
  seed=${seeds[$run]}
  python slurm_job_generator.py --model_out model_saves/${model_name}_basetags_${tagset_size}_${run} --wandb_project basetags_${tagset_size} --tagset_file tagsets/base/${tagset_size_literal}-labels.txt --data_dir datasets/preprocessed --seed $seed --base_model $model --model_outputs_dir model_dev_outputs
  python slurm_job_generator.py --model_out model_saves/${model_name}_spell_${tagset_size}_${run} --wandb_project spell_${tagset_size} --tagset_file tagsets/spell/${tagset_size_literal}-labels.txt --data_dir datasets/preprocessed --data_extension spell --seed $seed --base_model $model --model_outputs_dir model_dev_outputs
  python slurm_job_generator.py --model_out model_saves/${model_name}_lemon_${tagset_size}_${run} --wandb_project lemon_${tagset_size} --tagset_file tagsets/lemon/${tagset_size_literal}-labels.txt --data_dir datasets/preprocessed_fewer_transforms --data_extension lemon --seed $seed --base_model $model --model_outputs_dir model_dev_outputs
  python slurm_job_generator.py --model_out model_saves/${model_name}_lemon-spell_${tagset_size}_${run} --wandb_project lemon-spell_${tagset_size} --tagset_file tagsets/lemon_spell/${tagset_size_literal}-labels.txt --data_dir datasets/preprocessed_fewer_transforms --data_extension lemon.spell --seed $seed --base_model $model --model_outputs_dir model_dev_outputs
done

