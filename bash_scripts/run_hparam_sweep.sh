#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH -p <GPU PARTITION>

source venv/bin/activate

name=$(basename $1)
output_dir=$2

mkdir -p output_dir

if [ -d "${output_dir}/$name" ]; then rm -Rf "${output_dir}/$name"; fi  # https://stackoverflow.com/questions/4846007/check-if-directory-exists-and-delete-in-one-command-unix
mkdir -p "${output_dir}/$name"

# xargs -P 4 means run 4 processes in parallel (to saturate the GPU)
sh -c 'for t in 0.{00..90..2}; do for y in 0.{00..90..2}; do echo $t $y; done; done;' | \
xargs -n 1 -P 4 -L1 bash -c 'python predict.py --model_name_or_path "'$1'" --input_file dev_data/out_uncorr.dev.txt --output_file "'${output_dir}'/'$name'/${0}_${1}.txt" --additional_confidence $0 --min_error_probability $1'

# run 32 errant processes in parallel because we have 32 CPU cores
# the syntax ${0%.*} removes the ".txt" extension from the filename variable, $0
ls ${output_dir}/$name/*.txt | xargs -n 1 -P 32 -L1 bash -c 'errant_parallel -orig dev_data/out_uncorr.dev.txt -cor "$0" -out "${0%.*}.m2"'
ls ${output_dir}/$name/*.m2 | xargs -n 1 -P 32 -L1 bash -c 'errant_compare -hyp "$0" -ref dev_data/errant.m2 >> "${0%.*}.errant"'

python utils/errant_output_combiner.py --input_files ${output_dir}/$name/*.errant --output_file ${output_dir}/$name/summary.csv
python utils/hparam_csv_analysis.py --input_file ${output_dir}/$name/summary.csv --n_seeds 1 --copy_output_files True --create_json_files True

echo done
