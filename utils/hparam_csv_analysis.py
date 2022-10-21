import pandas as pd
import shutil
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate averages over multiple runs')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--visualise', type=bool, default=False, required=False)
parser.add_argument('--copy_output_files', type=bool, default=False, required=False)
parser.add_argument('--create_json_files', type=bool, default=False, required=False)
parser.add_argument('--model_saves_dir', type=str, default='model_saves', required=False)
parser.add_argument('--n_seeds', type=int, default=3, required=False)

args = parser.parse_args()

df = pd.read_csv(args.input_file, skipinitialspace=True)

# extract hparams from output file names
df2 = df.join(df['file'].str.extract(r'(?P<model>.*)/(?P<additional_confidence>[\.0-9]+)_(?P<min_error_probability>[\.0-9]+)(?:.errant)'))
df2 = df2.astype({'min_error_probability': float, 'additional_confidence': float})

# reduce to only the best hparams for each model
df3 = df2.loc[df2.reset_index().groupby(['model'])['F0.5'].idxmax()][['model', 'file', 'min_error_probability', 'additional_confidence', 'F0.5']]

# remove seed number and "_p3" from model name
df3['name'] = df3['model'].apply(lambda x: x[:-5])

# extract info from model name
df3 = df3.join(df3['model'].str.extract(r'.*/(?P<encoder>[\-a-z0-9]+)_(?P<tagset>[\-a-z]+)_(?P<tagset_size>[\-a-z0-9]+)_(?P<seed>.).*'))

assert (df3.groupby('name', sort=False).count() == args.n_seeds).all(axis=None), f"some of them don't have {args.n_seeds} runs"

if args.visualise:
    temp2 = df3[['encoder', 'tagset_size', 'tagset', 'F0.5']]
    temp2 = temp2.groupby(['encoder', 'tagset_size', 'tagset'], as_index=False).mean()
    temp2['encoder_tagset_size'] = temp2['encoder'] + '_' + temp2['tagset_size']
    temp2 = temp2.pivot_table(index=['encoder_tagset_size'], columns=['tagset'], values='F0.5')
    temp2 = temp2[['basetags', 'spell', 'lemon', 'lemon-spell']]

    temp2.rename(columns={'spell': '$SPELL', 'lemon': '$INFLECT', 'lemon-spell': '\$SPELL + \$INFLECT'}, inplace=True)

    from matplotlib import pyplot as plt
    ax = temp2.plot.bar(rot=45, figsize=(8, 5))
    plt.xticks(ha='right', rotation_mode='anchor')
    plt.ylim([0.555, 0.635])
    ax.axes.get_xaxis().get_label().set_visible(False)
    plt.ylabel('F0.5')
    plt.title(f'BEA-2019 dev scores (mean over {args.n_seeds} seeds)')
    plt.tight_layout()
    plt.show()

if args.copy_output_files:
    df3['output_file'] = df3['file'].apply(lambda x: x[:-6] + 'txt')
    df3['destination'] = df3['model'].apply(lambda x: x[:-3] + '.txt')
    for _, x in df3.iterrows():
        print('cp', x['output_file'], x['destination'])
        shutil.copyfile(x['output_file'], x['destination'])

if args.create_json_files:
    for _, r in df3[['model', 'min_error_probability', 'additional_confidence']].iterrows():
        json_file_name = args.model_saves_dir + '/' + os.path.basename(r['model']) + '/inference_tweak_params.json'
        print('creating', json_file_name)
        with open(json_file_name, 'w') as f:
            json.dump(r[['min_error_probability', 'additional_confidence']].to_dict(), f)
