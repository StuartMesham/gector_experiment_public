import os
import random
import string
import argparse

parser = argparse.ArgumentParser(description='Flatten and get True Positives only')
parser.add_argument('--template_dir', type=str, default='slurm_templates')
parser.add_argument('--script_output_dir', type=str, default='slurm/generated')
parser.add_argument('--p1_nodes', type=int, default=1)
parser.add_argument('--p2_nodes', type=int, default=1)
parser.add_argument('--p3_nodes', type=int, default=1)
parser.add_argument('--p1_gpus', type=int, default=4)
parser.add_argument('--p2_gpus', type=int, default=2)
parser.add_argument('--p3_gpus', type=int, default=1)
parser.add_argument('--model_out', type=str, required=True)
parser.add_argument('--model_outputs_dir', type=str, required=True, default='model_outputs', help='directory to save inference tuning outputs')
parser.add_argument('--wandb_project', type=str, required=True)
parser.add_argument('--tagset_file', type=str, required=True)
parser.add_argument('--slurm_out', type=str, default='slurm_outputs/%j.%x.out')
parser.add_argument('--slurm_err', type=str, default='slurm_outputs/%j.%x.err')
parser.add_argument('--data_extension', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--base_model', type=str, default='roberta-base')
parser.add_argument('--do_synthetic_pretrain', type=str, default='True')

args = parser.parse_args()

assert args.p1_nodes == args.p2_nodes == args.p3_nodes == 1, 'multi-node training no longer supported'

assert args.do_synthetic_pretrain in ['True', 'False']
args.do_synthetic_pretrain = args.do_synthetic_pretrain == 'True'

venv = 'venv'
partition = '<CLUSTER PARTITION>'
account = '<SLURM ACCOUNT>'
port = random.randint(12000, 13000)

assert args.data_dir is not None, 'please supply --data_dir argument'

if args.data_extension is not None:
    train_1 = f'{args.data_dir}/stage_1/train.{args.data_extension}.json'
    dev_1 = f'{args.data_dir}/stage_1/dev.{args.data_extension}.json'
    train_2 = f'{args.data_dir}/stage_2/train.{args.data_extension}.json'
    dev_2 = f'{args.data_dir}/stage_2/dev.{args.data_extension}.json'
    train_3 = f'{args.data_dir}/stage_3/train.{args.data_extension}.json'
    dev_3 = f'{args.data_dir}/stage_3/dev.{args.data_extension}.json'
else:
    train_1 = f'{args.data_dir}/stage_1/train.json'
    dev_1 = f'{args.data_dir}/stage_1/dev.json'
    train_2 = f'{args.data_dir}/stage_2/train.json'
    dev_2 = f'{args.data_dir}/stage_2/dev.json'
    train_3 = f'{args.data_dir}/stage_3/train.json'
    dev_3 = f'{args.data_dir}/stage_3/dev.json'

def generate_script(template_file, output_file, fill_values, delete_old_checkpoints=True):
    with open(template_file, 'r') as f:
        temp = f.read()

    if not delete_old_checkpoints:
        temp = temp.replace('find {{BASE_MODEL}} -type d -name "*checkpoint-*" | xargs rm -r', '')

    for k, v in fill_values.items():
        temp = temp.replace('{{' + k + '}}', '"' + str(v) + '"')
    with open(output_file, 'w') as f:
        f.write(temp)
    os.system(f'chmod 755 "{output_file}"')

assert 256 % (args.p1_nodes * args.p1_gpus) == 0
assert 128 % (args.p2_nodes * args.p2_gpus) == 0
assert 128 % (args.p3_nodes * args.p3_gpus) == 0

wandb_run_id_p1 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
wandb_run_id_p2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
wandb_run_id_p3 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

p1_values = {
    'BASE_MODEL': args.base_model,
    'TAGSET_FILE': args.tagset_file,
    'OUTPUT_DIR': f'{args.model_out}_p1',
    'BATCH_SIZE': 256//(args.p1_nodes * args.p1_gpus),
    'WANDB_RUN_ID': wandb_run_id_p1,
    'HF_HOME': f'/ramdisks/hf_cache/{wandb_run_id_p1}',
    'WANDB_PROJECT': args.wandb_project,
    'VENV': venv,
    'TRAIN_DATA': train_1,
    'DEV_DATA': dev_1,
    'SEED': args.seed,
    
    'JOB_NAME': wandb_run_id_p1,
    'N_NODES': args.p1_nodes,
    'N_GPUS': args.p1_gpus,
    'JOB_TIME': '12:00:00',
    'SLURM_OUTPUT_FILE': args.slurm_out,
    'SLURM_ERROR_FILE': args.slurm_err,
    'ACCOUNT': account,
    'PARTITION': partition,
    'MASTER_PORT': port,
}

p2_values = p1_values.copy()
p2_values.update({
    'BASE_MODEL': p1_values['OUTPUT_DIR'] if args.do_synthetic_pretrain else args.base_model,
    'IS_GECTOR_MODEL': args.do_synthetic_pretrain,
    'OUTPUT_DIR': f'{args.model_out}_p2',
    'BATCH_SIZE': 128//(args.p2_nodes * args.p2_gpus),
    'WANDB_RUN_ID': wandb_run_id_p2,
    'HF_HOME': f'/ramdisks/hf_cache/{wandb_run_id_p2}',
    'TRAIN_DATA': train_2,
    'DEV_DATA': dev_2,
    
    'JOB_NAME': wandb_run_id_p2,
    'N_NODES': args.p2_nodes,
    'N_GPUS': args.p2_gpus,
    'JOB_TIME': '12:00:00',
    'SLURM_OUTPUT_FILE': args.slurm_out,
    'SLURM_ERROR_FILE': args.slurm_err,
    'ACCOUNT': account,
    'PARTITION': partition,
    'MASTER_PORT': port,
})

p3_values = p2_values.copy()
p3_values.update({
    'BASE_MODEL': p2_values['OUTPUT_DIR'],
    'OUTPUT_DIR': f'{args.model_out}_p3',
    'BATCH_SIZE': 128//(args.p3_nodes * args.p3_gpus),
    'WANDB_RUN_ID': wandb_run_id_p3,
    'HF_HOME': f'/ramdisks/hf_cache/{wandb_run_id_p3}',
    'TRAIN_DATA': train_3,
    'DEV_DATA': dev_3,
    
    'JOB_NAME': wandb_run_id_p3,
    'N_NODES': args.p3_nodes,
    'N_GPUS': args.p3_gpus,
    'JOB_TIME': '01:00:00',
    'SLURM_OUTPUT_FILE': args.slurm_out,
    'SLURM_ERROR_FILE': args.slurm_err,
    'ACCOUNT': account,
    'PARTITION': partition,
    'MASTER_PORT': port,
})

script_1 = os.path.join(args.script_output_dir, f'{wandb_run_id_p1}.sh')
script_2 = os.path.join(args.script_output_dir, f'{wandb_run_id_p2}.sh')
script_3 = os.path.join(args.script_output_dir, f'{wandb_run_id_p3}.sh')

os.makedirs(args.script_output_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.slurm_out), exist_ok=True)
os.makedirs(os.path.dirname(args.slurm_err), exist_ok=True)

if args.do_synthetic_pretrain:
    generate_script(os.path.join(args.template_dir, 'train_p1_template.sh'), script_1, p1_values)
generate_script(os.path.join(args.template_dir, 'train_p2_template.sh'), script_2, p2_values, delete_old_checkpoints=args.do_synthetic_pretrain)
generate_script(os.path.join(args.template_dir, 'train_p3_template.sh'), script_3, p3_values)

if args.do_synthetic_pretrain:
    job_id_11 = os.popen(f'sbatch {script_1}').read().split()[-1]
    job_id_12 = os.popen(f'sbatch --dependency=afternotok:{job_id_11} {script_1}').read().split()[-1]
    job_id_2 = os.popen(f'sbatch --dependency=afterok:{job_id_12} {script_2}').read().split()[-1]
    job_id_3 = os.popen(f'sbatch --dependency=afterok:{job_id_2} {script_3}').read().split()[-1]
else:
    job_id_2 = os.popen(f'sbatch {script_2}').read().split()[-1]
    job_id_3 = os.popen(f'sbatch --dependency=afterok:{job_id_2} {script_3}').read().split()[-1]

os.popen(f'sbatch --dependency=afterok:{job_id_3} -J "tune {wandb_run_id_p3}" ./bash_scripts/run_hparam_sweep.sh {args.model_out}_p3 {args.model_outputs_dir}')

print('done')
