import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--f1', help='Path to file 1', required=True)
parser.add_argument('--f2', help='Path to file 2', required=True)
parser.add_argument('--split_names', default="['train', 'dev']")
parser.add_argument('--split_weights', default="[0.98, 0.02]")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)

input_file_names = [args.f1, args.f2]

split_names = eval(args.split_names)
split_weights = eval(args.split_weights)

input_files = [open(input_file_name, 'r') for input_file_name in input_file_names]

# output_files will be a 2d list of file handles indexed by [input_file_number][split_number]
output_files = []
for input_name in input_file_names:
    out_files = []
    for split_name in split_names:
        # generate output file name for this split
        if input_name.endswith('.txt'):
            out_file_name = f'{input_name[:-4]}.{split_name}.txt'  # e.g. file.txt -> file.dev.txt
        else:
            out_file_name = f'{input_name}.{split_name}'  # e.g. file -> file.dev
        out_files.append(open(out_file_name, 'w'))
    output_files.append(out_files)

while True:
    try:
        lines = [next(f) for f in input_files]
    except:
        break
    
    # randomly choose which split this sentence will go into
    split_index = random.choices(list(range(len(split_names))), split_weights)[0]

    for i, line in enumerate(lines):
        output_files[i][split_index].write(line)

# close all files
for f in input_files + [f for s in output_files for f in s]:  # flatten output files 2d list
    f.close()
