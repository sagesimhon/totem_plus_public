import argparse

import config
from distutils.util import strtobool

def bool_type(x):
    return bool(strtobool(x))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Totem Unwarping dataset_helpers')

    # Data paths
    parser.add_argument('--base_mappings_data_path', type=str, help='base path to mappings data', default=config.output_data_base_path)
    parser.add_argument('--exp_folder', type=str, required=True, help='name of mappings experiment folder, also where NN results will be saved')
    parser.add_argument('--run_extension', type=str, default='', help='identifier to separate different nn run experiments')

    # Dataset context
    parser.add_argument('--res_x', type=int, help='resolution of train dataset in x', required=True)
    parser.add_argument('--res_y', type=int, help='resolution of train dataset in y', required=True)
    parser.add_argument('--totem', type=str, default=None)

    # Manual datasets: Override automated dataset usage from exp folder:
    parser.add_argument('--is_manual_testset', type=bool_type, default=False, help='override torch Dataset creation for test/val data')
    parser.add_argument('--manual_testset_inputs', type=str, default=None, help='path to manual test/val torch.utils.data.Dataset object')
    parser.add_argument('--manual_testset_outputs', type=str, default=None, help='path to manual test/val torch.utils.data.Dataset object')

    parser.add_argument('--hard_tots', type=str, default=None, help='manual dataset labels')
    parser.add_argument('--hard_cams', type=str, default=None, help='manual dataset inputs')
    parser.add_argument('--hard_tots_comp', type=str, default=None, help='manual dataset held out labels')
    parser.add_argument('--hard_cams_comp', type=str, default=None, help='manual dataset held out inputs')

    # Model architecture
    parser.add_argument('--d', type=int, help='depth of model', default=8)
    parser.add_argument('--w', type=int, help='width of model', default=256)

    # Training
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--use_embed_fn', type=bool_type, default=True, help='use positional encoding')
    parser.add_argument('--n_freqs', type=int, default=10, help='number of frequencies for positional encoding')

    args = parser.parse_args()
    return args