import os
import json
import argparse
from multiprocessing import Pool

from symspellpy import Verbosity

from gector_utils import load_sym_spell
from helpers_new import convert_using_case
from utils.preprocess_data import check_casetype

max_edit_distance = int(os.environ.get('SPELL2_PREPROCESS_MAX_DISTANCE', 2))
print('max_edit_distance:', max_edit_distance)

sym_spell = load_sym_spell('frequency_dictionary_en_82_765.txt', max_edit_distance=max_edit_distance)


def add_inflect_tokens(line):
    j = json.loads(line)

    for i, (token, label) in enumerate(zip(j['tokens'], j['labels'])):
        if label.startswith('$REPLACE'):
            target = label.split('_', 1)[1]
            case_type = check_casetype(token.lower(), token)

            suggestions = [s.term for s in sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=max_edit_distance, transfer_casing=case_type is None)]

            if len(suggestions) == 0:
                suggestion = None
            else:
                suggestion = suggestions[0]
                if case_type is not None:
                    suggestion = convert_using_case(suggestion, case_type)

            if suggestion == token:  # don't use the SPELL_2 tag to skip capitalization corrections
                continue

            if suggestion is not None and check_casetype(suggestion, target) is not None:
                j['labels'][i] = "$SPELL_2"

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
