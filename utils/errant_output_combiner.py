import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create CSV from errant outputs')
parser.add_argument('--input_files', nargs='+', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()

HEADERS = ['file', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F0.5']

with open(args.output_file, 'w') as out_file:
    out_file.write(', '.join(HEADERS) + '\n')

    for input_file in tqdm(args.input_files):
        with open(input_file, 'r') as in_file:
            next(in_file)
            assert 'Span-Based Correction' in next(in_file)
            assert next(in_file).split() == HEADERS[1:]
            values = [input_file] + next(in_file).split()
            assert len(values) == 7
            out_file.write(', '.join(values) + '\n')
