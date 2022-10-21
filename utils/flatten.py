import json
import argparse

parser = argparse.ArgumentParser(description='Flatten and get True Positives only')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--tp_only', type=bool, default=False)


args = parser.parse_args()

assert args.input_file.endswith('.json')

print('processing', args.input_file)

with open(args.input_file, 'r') as in_file, open(args.output_file, 'w') as out_file:
    for line in in_file:
        j = json.loads(line)
        j['labels'] = [l[0] for l in j['labels']]
        if args.tp_only:
            is_tp = False
            # for label in [l[0] for l in j['labels']]:
            for label in j['labels']:
                if label != '$KEEP':
                    is_tp = True
                    break
            if is_tp:
                out_file.write(json.dumps(j) + '\n')
        else:
            out_file.write(json.dumps(j) + '\n')
