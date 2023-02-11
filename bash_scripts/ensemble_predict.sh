#!/bin/bash

# customise the variables below

input_file=$1
output_file=$2
temp_output_dir=temp_ensemble_outputs

e1=deberta-large
es1=5k

e2=electra-large
es2=5k

e3=roberta-large
es3=5k

tagset=lemon-spell
run=2

mkdir -p $temp_output_dir

python predict.py --model_name_or_path stuartmesham/${e1}_${tagset}_${es1}_${run}_p3 --input_file ${input_file} --output_file $temp_output_dir/${e1}_${tagset}_${es1}_${run}.txt
python predict.py --model_name_or_path stuartmesham/${e2}_${tagset}_${es2}_${run}_p3 --input_file ${input_file} --output_file $temp_output_dir/${e2}_${tagset}_${es2}_${run}.txt
python predict.py --model_name_or_path stuartmesham/${e3}_${tagset}_${es3}_${run}_p3 --input_file ${input_file} --output_file $temp_output_dir/${e3}_${tagset}_${es3}_${run}.txt

python utils/ensemble.py --source_file ${input_file} --target_files $temp_output_dir/${e1}_${tagset}_${es1}_${run}.txt $temp_output_dir/${e2}_${tagset}_${es2}_${run}.txt $temp_output_dir/${e3}_${tagset}_${es3}_${run}.txt --output_file $output_file

echo 'done'
