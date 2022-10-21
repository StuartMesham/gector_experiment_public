import argparse

parser = argparse.ArgumentParser()
parser.add_argument("m2_file", help="The path to an input m2 file.")
parser.add_argument("-out", help="A path to where we save the output text file.", required=True)
args = parser.parse_args()

with open(args.m2_file, 'r') as infile, open(args.out, 'w') as outfile:
  for line in infile:
    parts = line.split()
    if len(parts) == 0:
      continue
    elif parts[0] == 'S':
      outfile.write(' '.join(parts[1:]) + '\n')
