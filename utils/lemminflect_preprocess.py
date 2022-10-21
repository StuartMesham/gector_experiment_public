import os
import json
import argparse
import spacy
import lemminflect
from multiprocessing import Pool

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = nlp.tokenizer.tokens_from_list


def add_inflect_tokens(line):
    j = json.loads(line)
    source_tokens = j['tokens'][1:]
    doc = nlp(source_tokens)

    # create copy of sentence with all $REPLACE_{} operations applied
    target_tokens = source_tokens.copy()
    for i, label in enumerate(j['labels'][1:]):
        if label.startswith('$REPLACE'):
            target_tokens[i] = label.split('_', 1)[1]

    target_pos_tags = [t.tag_ for t in nlp(target_tokens)]

    for i, label in enumerate(j['labels']):
        if label.startswith('$REPLACE_'):
            try:
                if doc[i - 1]._.inflect(target_pos_tags[i - 1]) == label.split('_', 1)[1]:
                    j['labels'][i] = f'$INFLECT_{target_pos_tags[i - 1]}'
            except Exception:
                print('failed sentence:', source_tokens)

    return json.dumps(j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    slurm_allowed_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_allowed_cpus is not None:
        slurm_allowed_cpus = int(slurm_allowed_cpus)

    with Pool(slurm_allowed_cpus) as p:
        processed_lines = p.map(add_inflect_tokens, lines)

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(processed_lines) + '\n')
