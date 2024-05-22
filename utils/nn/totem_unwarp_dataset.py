import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.nn import utils as utils
import torch
import json
import numpy as np

def load_data(filename, is_rev=False):
    with open(filename, 'r') as f:
        mappings = json.load(f)

    uv_tots = []
    uv_cams = []
    if not is_rev:
        for uv_tot, uv_cam in mappings.items():
            # Convert the coordinates from strings to tuples
            uv_tot = [float(num) for num in uv_tot.strip('()').split(',')]   # [x,y]
            # TODO shift these for visualization (see shift_uv_tots in sages_color_corr_messy.py)
            uv_cam = [float(num) for num in uv_cam.strip('()').split(',')]   # [x,y]
            uv_tots.append(uv_tot)
            uv_cams.append(uv_cam)
    else:
        for uv_cam, uv_tot in mappings.items():
            # Convert the coordinates from strings to tuples
            uv_tot = [float(num) for num in uv_tot.strip('()').split(',')]   # [x,y]
            # TODO shift these for visualization (see shift_uv_tots in sages_color_corr_messy.py)
            uv_cam = [float(num) for num in uv_cam.strip('()').split(',')]   # [x,y]
            uv_tots.append(uv_tot)
            uv_cams.append(uv_cam)

    return uv_tots, uv_cams

class TotemUnwarpDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, transform=None):
        super(TotemUnwarpDataset, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        data, label = self.inputs[idx], self.outputs[idx]
        return data, label


def get_train_val_dataset(data_path, ratio=0.80, random_seed=0, with_outlier=True, rev_mappings=True, hardcoded=False,
                          hardcoded_path_tots=None, hardcoded_path_cams=None, hardcoded_complement_tots=None,
                          hardcoded_complement_cams=None, res=None, save=False
                          ):
    np.random.seed(random_seed)

    if with_outlier:
        path_to_mappings = os.path.join(data_path, 'mappings_with_outlier.json')
    else:
        path_to_mappings = os.path.join(data_path, 'mappings.json')
    path_to_rev_mappings = os.path.join(data_path, 'mappings_cam_to_tot.json')
    if rev_mappings:
        path_to_mappings = path_to_rev_mappings

    uv_tots_full, uv_cams_full = load_data(path_to_mappings,
                                           is_rev=rev_mappings)  # uv_cams is the input, uv_tots is the target

    # uv_tots_full = torch.Tensor(np.array(uv_tots_full))
    # uv_cams_full = torch.Tensor(np.array(uv_cams_full))
    # mins_i = uv_cams_full.min(dim=0)[0]
    # maxs_i = uv_cams_full.max(dim=0)[0]
    # mins_o = uv_tots_full.min(dim=0)[0]
    # maxs_o = uv_tots_full.max(dim=0)[0]

    val_len = int(len(uv_cams_full) * (1-ratio))
    fixed_val_len = int(len(uv_cams_full) * (1-0.8)) # 11108 (for 256) = 20% val/test
    if hardcoded:
        uv_cams_custom = np.load(hardcoded_path_cams)
        uv_tots_custom = np.load(hardcoded_path_tots)

        complement_uv_cams = np.load(hardcoded_complement_cams)
        complement_uv_tots = np.load(hardcoded_complement_tots)

        # shuffle uv_cams and uv_tots in the same way
        shuffled_indices = np.random.permutation(len(uv_cams_custom))
        uv_cams_custom = [uv_cams_custom[i] for i in shuffled_indices]
        uv_tots_custom = [uv_tots_custom[i] for i in shuffled_indices]

        # now shuffle uv_cams and uv_tots in complement
        shuffled_indices = np.random.permutation(len(complement_uv_cams))
        complement_uv_cams = [complement_uv_cams[i] for i in shuffled_indices]
        complement_uv_tots = [complement_uv_tots[i] for i in shuffled_indices]

        if not rev_mappings:
            inputs = torch.Tensor(np.array(uv_tots_custom))
            outputs = torch.Tensor(np.array(uv_cams_custom))
            vals_inputs = torch.Tensor(np.array(complement_uv_tots))
            vals_outputs = torch.Tensor(np.array(complement_uv_cams))
        else:
            inputs = torch.Tensor(np.array(uv_cams_custom))
            outputs = torch.Tensor(np.array(uv_tots_custom))
            vals_inputs = torch.Tensor(np.array(complement_uv_cams))
            vals_outputs = torch.Tensor(np.array(complement_uv_tots))

            # Stack the tensors to get a (n+m)x2 tensor
            inputs_and_val_complement = torch.cat((inputs, vals_inputs), dim=0) # should be size of og dataset from gen
            outputs_and_val_complement = torch.cat((outputs, vals_outputs), dim=0)

            normalized_inputs_and_vals, in_mins, in_maxs = utils.normalize(inputs_and_val_complement)
            normalized_outputs_and_vals, out_mins, out_maxs = utils.normalize(outputs_and_val_complement)

            # Split them back into original tensors
            train_inputs, vals_inputs = normalized_inputs_and_vals.split([inputs.size(0), vals_inputs.size(0)], dim=0)
            train_outputs, vals_outputs = normalized_outputs_and_vals.split([outputs.size(0), vals_outputs.size(0)], dim=0)
            #train_inputs, in_mins, in_maxs = utils.normalize(inputs, mins=mins_i, maxs=maxs_i)
            #train_outputs, out_mins, out_maxs = utils.normalize(outputs, mins=mins_o, maxs=maxs_o)

            # vals_inputs, _, _ = utils.normalize(vals_inputs, mins=mins_i, maxs=maxs_i)
            # vals_outputs, _, _ = utils.normalize(vals_outputs, mins=mins_o, maxs=maxs_o)

            # For fair comparisons, take a random sample of fixed size of val_inputs/val_outputs
            vals_inputs = vals_inputs[:fixed_val_len]
            vals_outputs = vals_outputs[:fixed_val_len]

            print(f"LEN OF VAL SET (PRE TEST/VAL SPLIT): {len(vals_inputs)}")
            print(f"LEN OF TRAIN SET: {len(train_inputs)}")

    else:
        if not rev_mappings:
            inputs = uv_tots_full
            outputs = uv_cams_full
        else:
            inputs = uv_cams_full
            outputs = uv_tots_full


        # shuffle uv_cams and uv_tots in the same way
        shuffled_indices = np.random.permutation(len(inputs))
        inputs = [inputs[i] for i in shuffled_indices]
        inputs = torch.Tensor(np.array(inputs))
        outputs = [outputs[i] for i in shuffled_indices]
        outputs = torch.Tensor(np.array(outputs))

        inputs, in_mins, in_maxs = utils.normalize(inputs)#, mins=mins_i, maxs=maxs_i)
        outputs, out_mins, out_maxs = utils.normalize(outputs)#, mins=mins_o, maxs=maxs_o)

        # length_dataset = len(inputs)
        # train_length = int(length_dataset * ratio)
        #
        # train_inputs = inputs[:train_length]
        # train_outputs = outputs[:train_length]
        # vals_inputs = inputs[train_length:]
        # vals_outputs = outputs[train_length:]
        vals_inputs = inputs[:val_len]
        vals_outputs = outputs[:val_len]
        vals_inputs = vals_inputs[:fixed_val_len]
        vals_outputs = vals_outputs[:fixed_val_len]

        train_inputs = inputs[val_len:]
        train_outputs = outputs[val_len:]

        print(f"LEN OF VAL SET (PRE TEST/VAL SPLIT): {len(vals_inputs)}")
        print(f"LEN OF TRAIN SET: {len(train_inputs)}")

    train_dataset = TotemUnwarpDataset(train_inputs, train_outputs)
    val_dataset = TotemUnwarpDataset(vals_inputs, vals_outputs)

    if save:
        torch.save(vals_inputs, f'NEW_{res}_testset_in.pt')
        torch.save(vals_outputs, f'NEW_{res}_testset_out.pt')
        print(f"{res} IN MINS, IN MAXS, OUT MINS, OUT MAXS: {in_mins}, {in_maxs}, {out_mins}, {out_maxs}")

    return train_dataset, val_dataset, in_mins, in_maxs, out_mins, out_maxs


# def get_train_val_dataset(data_path, ratio=0.80, random_seed=0, with_outlier=True, rev_mappings=True, hardcoded=False, hardcoded_path_tots=None, hardcoded_path_cams=None, hardcoded_complement_tots=None, hardcoded_complement_cams=None
#                           ):
#     np.random.seed(random_seed)
#
#     if with_outlier:
#         path_to_mappings = os.path.join(data_path, 'mappings_with_outlier.json')
#     else:
#         path_to_mappings = os.path.join(data_path, 'mappings.json')
#     path_to_rev_mappings = os.path.join(data_path, 'mappings_cam_to_tot.json')
#     if rev_mappings:
#         path_to_mappings = path_to_rev_mappings
#
#     uv_tots_full, uv_cams_full = load_data(path_to_mappings, is_rev=rev_mappings) # uv_cams is the input, uv_tots is the target
#     uv_tots_full = torch.Tensor(np.array(uv_tots_full))
#     uv_cams_full = torch.Tensor(np.array(uv_cams_full))
#
#     uv_tots, uv_cams = uv_tots_full.copy(), uv_cams_full.copy()
#     complement_uv_cams = uv_cams_full
#     complement_uv_tots = uv_tots_full
#
#     if hardcoded:
#         uv_cams = np.load(hardcoded_path_cams)
#         uv_tots = np.load(hardcoded_path_tots)
#
#         complement_uv_cams = np.load(hardcoded_complement_cams)
#         complement_uv_tots = np.load(hardcoded_complement_tots)
#
#
#     # TODO also hardcode in the test set
#     # '/Users/sage/PycharmProjects/totem_new_shapes/uv_cams_512_75pruned.np.npy'
#
#     # shuffle uv_cams and uv_tots in the same way
#     shuffled_indices = np.random.permutation(len(uv_cams))
#     uv_cams = [uv_cams[i] for i in shuffled_indices]
#     uv_tots = [uv_tots[i] for i in shuffled_indices]
#
#
#     # now shuffle uv_cams and uv_tots in complement
#     shuffled_indices = np.random.permutation(len(complement_uv_cams))
#     complement_uv_cams = [complement_uv_cams[i] for i in shuffled_indices]
#     complement_uv_tots = [complement_uv_tots[i] for i in shuffled_indices]
#
#     if not rev_mappings:
#         inputs = torch.Tensor(np.array(uv_tots))
#         outputs = torch.Tensor(np.array(uv_cams))
#         vals_inputs = torch.Tensor(np.array(complement_uv_tots))
#         vals_outputs = torch.Tensor(np.array(complement_uv_cams))
#     else:
#         inputs = torch.Tensor(np.array(uv_cams))
#         outputs = torch.Tensor(np.array(uv_tots))
#         vals_inputs = torch.Tensor(np.array(complement_uv_cams))
#         vals_outputs = torch.Tensor(np.array(complement_uv_tots))
#
#     mins_i = uv_cams_full.min(dim=0)[0]
#     maxs_i = uv_cams_full.max(dim=0)[0]
#     mins_o = uv_cams_full.min(dim=0)[0]
#     maxs_o = uv_cams_full.max(dim=0)[0]
#
#     inputs, in_mins, in_maxs = utils.normalize(inputs, mins=mins_i, maxs=maxs_i)
#     outputs, out_mins, out_maxs = utils.normalize(outputs, mins=mins_o, maxs=maxs_o)
#     val_inputs, _, _ = utils.normalize(vals_inputs, mins=mins_i, maxs=maxs_i)
#     val_outputs, _, _ = utils.normalize(vals_outputs, mins=mins_o, maxs=maxs_o)
#
#     length_dataset = len(inputs)
#     train_length = int(length_dataset * ratio)
#
#     train_inputs = inputs[:train_length]
#     train_outputs = outputs[:train_length]
#
#     if not hardcoded:
#         val_inputs = inputs[train_length:]
#         val_outputs = outputs[train_length:]
#
#     else:
#         val_outputs, val_inputs = complement_uv_tots, complement_uv_cams
#
#     train_dataset = TotemUnwarpDataset(train_inputs, train_outputs)
#     val_dataset = TotemUnwarpDataset(val_inputs, val_outputs)
#
#     return train_dataset, val_dataset, in_mins, in_maxs, out_mins, out_maxs