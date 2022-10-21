#!/bin/bash

# customise the variables below

input_file=data_downloads/wi+locness/test/ABCN.test.bea19.orig
output_dir=ensemble_outputs

e1=deberta-large
es1=5k

e2=electra-large
es2=5k

e3=roberta-large
es3=5k

mkdir -p $output_dir

for tagset in basetags spell lemon lemon-spell
do
  for run in 1 2 3 4 5 6
  do
    python predict.py --model_name_or_path model_saves/${e1}_${tagset}_${es1}_${run}_p3 --input_file ${input_file} --output_file $output_dir/${e1}_${tagset}_${es1}_${run}.txt
    python predict.py --model_name_or_path model_saves/${e2}_${tagset}_${es2}_${run}_p3 --input_file ${input_file} --output_file $output_dir/${e2}_${tagset}_${es2}_${run}.txt
    python predict.py --model_name_or_path model_saves/${e3}_${tagset}_${es3}_${run}_p3 --input_file ${input_file} --output_file $output_dir/${e3}_${tagset}_${es3}_${run}.txt

    python utils/ensemble.py --source_file ${input_file} --target_files $output_dir/${e1}_${tagset}_${es1}_${run}.txt $output_dir/${e2}_${tagset}_${es2}_${run}.txt $output_dir/${e3}_${tagset}_${es3}_${run}.txt --output_file $output_dir/ensemble_${tagset}_${run}.txt
  done
done

echo 'done'
