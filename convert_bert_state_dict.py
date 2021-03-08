import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--input_checkpoint', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()

state_dict = torch.load(args.input_checkpoint, map_location='cpu')['model_dict']

old_keys = []
new_keys = []
del_keys = []
for key in state_dict.keys():
    if 'question_model' in key:
        old_keys.append(key)
        new_keys.append(key.replace('question_model', 'bert'))
    if 'ctx_model' in key:
        del_keys.append(key)

for old_key, new_key in zip(old_keys, new_keys):
    state_dict[new_key] = state_dict.pop(old_key)
for del_key in del_keys:
    state_dict.pop(del_key)

torch.save(state_dict, args.output_file)
