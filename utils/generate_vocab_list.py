import json
import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser(description='Generate vocab file based on min frequency')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True, help='The substring {n} will be replaced with the vocab size')
parser.add_argument('--min_freq', type=int, required=False)
parser.add_argument('--vocab_size', type=int, required=False)

args = parser.parse_args()

if args.min_freq is not None:
    assert args.vocab_size is None

if args.vocab_size is not None:
    assert args.min_freq is None

rows = []
with open(args.input_file, 'r') as f:
    for i, line in enumerate(f):
        j = json.loads(line)
        for token, label in zip(j['tokens'], j['labels']):
            rows.append({'sentence': i, 'token': token, 'label': label})

df2 = pd.DataFrame(rows)

temp = df2.groupby(['label']).size().sort_values(ascending=False)

if args.min_freq is not None:
    vocab = list(temp.loc[temp >= args.min_freq].index)
else:
    assert args.vocab_size is not None
    vocab = list(temp[:args.vocab_size - 1].index)

print('vocab_size:', len(vocab) + 1)

output_file_name = args.output_file.replace('{n}', str(len(vocab) + 1))

os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
with open(output_file_name, 'w') as f:
    f.write('\n'.join(vocab + ['@@UNKNOWN@@']))
