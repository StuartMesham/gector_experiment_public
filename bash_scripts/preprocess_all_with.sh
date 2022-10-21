#!/bin/bash

# This script takes as input a preprocessing script and applies it to the datasets all three phases
# The output folders are created under the output directory supplied as argument 2

# Arguments:
#1 preprocess script e.g. utils/preprocess_data.py
#2 output dir

in_dir=unprocessed
out_dir=$2

print_and_run () {
  echo "$1"
  eval "$1"
}

for stage in 1 2 3
do
  mkdir -p $out_dir/stage_$stage
  for split in train dev
  do

    print_and_run "python $1 -s $in_dir/stage_$stage/out_uncorr.$split.txt -t $in_dir/stage_$stage/out_corr.$split.txt -o $out_dir/stage_$stage/$split.unshuffled.unflattened.json"

    if [ $stage = 3 ]
    then
      print_and_run "python utils/flatten.py --input_file $out_dir/stage_$stage/$split.unshuffled.unflattened.json --output_file $out_dir/stage_$stage/$split.unshuffled.json"
    else
      print_and_run "python utils/flatten.py --input_file $out_dir/stage_$stage/$split.unshuffled.unflattened.json --output_file $out_dir/stage_$stage/$split.unshuffled.json --tp_only True"
    fi

    print_and_run "shuf $out_dir/stage_$stage/$split.unshuffled.json > $out_dir/stage_$stage/$split.json"

  done
done
