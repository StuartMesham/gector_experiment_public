# An Extended Sequence Tagging Vocabulary for Grammatical Error Correction

Throughout our codebase we use the terms "spell" and "lemon".
They refer to spelling and lemminflect-related things respectively.

## Some external file sources
`utils/corr_from_m2.py` was downloaded from [this link](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/corr_from_m2.py)

`ensemble.py` was downloaded from [Tarnyavski et al.'s repository](https://github.com/MaksTarnavskyi/gector-large/blob/master/ensemble.py)

## Installation
This implementation was developed on Python 3.8

### macOS
Note for macOS users, some dependencies are not compatible with the python interpreter pre-installed on macOS.
Please use a version installed with [pyenv](https://github.com/pyenv/pyenv), [Homebrew](https://brew.sh) or downloaded from [the official Python website](https://www.python.org).
This repo was tested on python 3.8.13 installed with pyenv.

Local install command:
```bash
pip install -r requirements.txt
python -m spacy download en
```

### Example linux SLURM-based GPU server install
Tested with Python 3.8.2
```bash
module load python/3.8
python -m venv venv
module unload python/3.8
source venv/bin/activate
pip install --upgrade pip
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m spacy download en
```

For reference, we have included the `long-requirements.txt` file which has the versions of all python packages installed on our server when running our experiments.

## Export PYTHONPATH
Do this at the start of each terminal session
```bash
export PYTHONPATH="${PYTHONPATH}:`pwd`"
```

## Data downloading and initial splitting
Note that the datasets used are not all publicly available without requesting permission from the owners first.
In `bash_scripts/download_and_combine_data.sh` the command for downloading NUCLE has been removed.
Please request the dataset from the owners and add your own download command to the script.

This takes ~5.5 minutes on Google Colab:
```bash
./bash_scripts/download_and_combine_data.sh
```

This creates a temporary directory, `data_downloads`, and an output directory, `datasets/unprocessed`.
Our preprocessing scripts will read the data from `datasets/unprocessed`.

## Omelianchuk et al. Preprocessing

The following commands apply [Omelianchuk et al.'s preprocessing code](https://github.com/grammarly/gector/blob/master/utils/preprocess_data.py)
The second one has their inflection tags removed (see our paper's "preprocessing" section)

This takes ~1.5 hours
```bash
./bash_scripts/preprocess_all_with.sh utils/preprocess_data.py datasets/preprocessed
./bash_scripts/preprocess_all_with.sh utils/preprocess_data_fewer_transforms.py datasets/preprocessed_fewer_transforms
```

These commands create the `datasets/preprocessed` and `datasets/preprocessed_fewer_transforms` directories.

## Preprocessing for our new tags

The `utils/spelling_preprocess.py` and `utils/lemminflect_preprocess.py` scripts attempt to change `$REPLACE_{t}` tags to `$SPELL` and `$INFLECT_{POS}` tags respectively (see our paper's "preprocessing" section).

Note that these scripts support multiprocessing.
The `utils/lemminflect_preprocess.py` is computationally intensive and should be run on many cores.
On our system `utils/lemminflect_preprocess.py` took ~35 minutes on 76 cores.

```bash
./bash_scripts/tag_preprocess_all_with.sh utils/spelling_preprocess.py datasets/preprocessed json spell.json

./bash_scripts/tag_preprocess_all_with.sh utils/lemminflect_preprocess.py preprocessed_fewer_transforms json lemon.json
./bash_scripts/tag_preprocess_all_with.sh utils/spelling_preprocess.py preprocessed_fewer_transforms lemon.json lemon.spell.json
```

The datasets directory should now look like this:
```
datasets
|-- unprocessed
|   `-- ...
|-- preprocessed
|   |-- stage_1
|   |   |-- dev.json
|   |   |-- dev.spell.json
|   |   |-- train.json
|   |   `-- train.spell.json
|   |-- stage_2
|   |   |-- dev.json
|   |   |-- dev.spell.json
|   |   |-- train.json
|   |   `-- train.spell.json
|   `-- stage_3
|       |-- dev.json
|       |-- dev.spell.json
|       |-- train.json
|       `-- train.spell.json
`-- preprocessed_fewer_transforms
    |-- stage_1
    |   |-- dev.json
    |   |-- dev.lemon.json
    |   |-- dev.lemon.spell.json
    |   |-- train.json
    |   |-- train.lemon.json
    |   `-- train.lemon.spell.json
    |-- stage_2
    |   |-- dev.json
    |   |-- dev.lemon.json
    |   |-- dev.lemon.spell.json
    |   |-- train.json
    |   |-- train.lemon.json
    |   `-- train.lemon.spell.json
    `-- stage_3
        |-- dev.json
        |-- dev.lemon.json
        |-- dev.lemon.spell.json
        |-- train.json
        |-- train.lemon.json
        `-- train.lemon.spell.json
```

Files with `.spell` in the extension have had some replace tags changed to `$SPELL` tags.
Files with `.lemon` in the extension have had some replace tags changed to `$INFLECT_{POS}` tags.
The files in the `preprocessed_fewer_transforms` folder do not contain Omelianchuk et al.'s inflection-related tags.

## Generate vocabularies

The 5k and 10k vocabularies have 5000 and 10000 tags each +1 OOV tag.

Generate 5k vocab files:
```bash
python utils/generate_vocab_list.py --input_file datasets/preprocessed/stage_2/train.json --output_file tagsets/base/{n}-labels.txt --vocab_size 5001
python utils/generate_vocab_list.py --input_file datasets/preprocessed/stage_2/train.spell.json --output_file tagsets/spell/{n}-labels.txt --vocab_size 5001
python utils/generate_vocab_list.py --input_file datasets/preprocessed_fewer_transforms/stage_2/train.lemon.json --output_file tagsets/lemon/{n}-labels.txt --vocab_size 5001
python utils/generate_vocab_list.py --input_file datasets/preprocessed_fewer_transforms/stage_2/train.lemon.spell.json --output_file tagsets/lemon_spell/{n}-labels.txt --vocab_size 5001
```

Generate 10k vocab files:
```bash
python utils/generate_vocab_list.py --input_file datasets/preprocessed/stage_2/train.json --output_file tagsets/base/{n}-labels.txt --vocab_size 10001
python utils/generate_vocab_list.py --input_file datasets/preprocessed/stage_2/train.spell.json --output_file tagsets/spell/{n}-labels.txt --vocab_size 10001
python utils/generate_vocab_list.py --input_file datasets/preprocessed_fewer_transforms/stage_2/train.lemon.json --output_file tagsets/lemon/{n}-labels.txt --vocab_size 10001
python utils/generate_vocab_list.py --input_file datasets/preprocessed_fewer_transforms/stage_2/train.lemon.spell.json --output_file tagsets/lemon_spell/{n}-labels.txt --vocab_size 10001
```

## Train models

A single stage of training can be performed using the `run_gector.py` script.
This script is based on, and takes mostly the same arguments as HuggingFace's [run_ner.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py) script.
The script can be used with encoders of any huggingface architecture.

When loading from a checkpoint that is not a gector model (e.g. initialising from a pretrained LM checkpoint), supply the argument `--is_gector_model False`.
When loading from a checkpoint that is a gector model (e.g. when starting training stage 2 or 3 and loading the model checkpoint trained on the previous stage) use `--is_gector_model True`.

Example use:

```bash
python run_gector.py \
--do_train True \
--do_eval True \
--do_predict False \
--model_name_or_path microsoft/deberta-large \
--is_gector_model False \
--train_file datasets/stage_1/train.json \
--validation_file datasets/stage_1/dev.json \
--tagset_file tagsets/base/5001-labels.txt \
--label_column_name labels \
--lr_scheduler_type constant \
--output_dir model_saves/deberta-large_basetags_10k_42_p1 \
--early_stopping_patience 3 \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--evaluation_strategy steps \
--save_strategy steps \
--save_total_limit 1 \
--max_steps 200000 \
--eval_steps 10000 \
--save_steps 10000 \
--logging_steps 1000 \
--cold_steps 20000 \
--cold_lr 1e-3 \
--learning_rate 1e-5 \
--optim adamw_torch \
--classifier_dropout 0.0 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--weight_decay 0 \
--report_to wandb_resume \
--wandb_run_id test_run_1 \
--seed 42 \
--dataloader_num_workers 1
```

## Run Inference

Here is an example using our inference script.
If 
The inference tweak parameter arguments, `--additional_confidence` and `--min_error_probability` can be omitted if `model_save_dir` contains an `inference_tweak_params.json` file.

```bash
python predict.py --model_name_or_path <model_save_dir> --input_file <input.txt> --output_file <output.txt>
```

The command above expects `model_save_dir` to contain a file named `inference_tweak_params.json` containing json in the following format:
```json
{"min_error_probability": 0.64, "additional_confidence": 0.36}
```

You need to manually create this file after training with `run_gector.py`.
If you do not have an `inference_tweak_params.json` file, you can manually supply the `--additional_confidence` and `--min_error_probability` arguments to the `predict.py` script.

Our pipeline (shown below) uses a SLURM job script (`bash_scripts/run_hparam_sweep.sh`) to create `inference_tweak_params.json` files.
It performs a grid search over the two hyperparameters on the BEA-2019 dev set. 

## Run Ensembling

The `ensemble.py` script is documented [here](https://github.com/MaksTarnavskyi/gector-large).

## Model saves

Our model saves (which all include `inference_tweak_params.json` files) can be downloaded from (LINK REMOVED FOR BLIND REVIEW)

## Our Training Pipeline

We have included some examples of scripts we used to train the models in our paper.
These scripts were designed to run specifically on our SLURM-based cluster so will not out-of-the-box on a local machine or VM.
They have also been anonymised for release - we have removed commands using sensitive file paths and account names from the files.

```bash
for encoder in roberta-large xlnet-large-cased microsoft/deberta-large microsoft/deberta-v3-large google/electra-large-discriminator
do
  for vocab_size in 5k 10k
  do
    ./bash_scripts/train_all_seeds_for_model.sh ${encoder} ${vocab_size}
  done
done
```

The script above ultimately performs many runs of the `utils/slurm_job_generator.py` script.
The `utils/slurm_job_generator.py` script generates and submits a series of SLURM job scripts using the templates in `slurm_templates` and using `bash_scripts/run_hparam_sweep.sh`
These SLURM jobs perform all three phases of training and run the hparam tuning to `inference_tweak_params.json` files for each model save. 