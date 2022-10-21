#!/bin/bash

# Arguments
#1 tag preprocessing script
#2 dataset directory
#3 input extension (e.g. "json" or "lemon.json")
#4 extension to add (e.g. "spell.json" or "lemon.json")

for stage in 1 2 3 
do
  for split in train dev
    do
    python $1 --input_file ${2}/stage_$stage/$split.${4} --output_file ${2}/stage_$stage/$split.${4}
  done
done
