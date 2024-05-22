import argparse
from config import input_data_base_path, output_data_base_path
from distutils.util import strtobool


def bool_type(x):
    return bool(strtobool(x))


def parse_mappings_arguments():
    parser = argparse.ArgumentParser(description='Totem Mapping dataset_gen_helpers')

    # Data paths
    parser.add_argument('--input_data_base_path', type=str, default=input_data_base_path, help='base path to input data (required xml files for generation)')
    parser.add_argument('--output_data_base_path', type=str, default=output_data_base_path, help='path to output data, which holds all experiments. Can live anywhere, including outside the project directory.')
    parser.add_argument('--exp_folder', type=str, required=True, help='name you would like to give this experiment', default='MyExperiment')

    # Settings (sweep granularity, resolution, rendering setting, parallelization settings)
    parser.add_argument('--res', type=int, required=True, help='desired resolution for desired mapping generation. Supported options are 728, 256 at the moment')
    parser.add_argument('--n', type=int, default=999, help='granularity of the sweep. Max 999 means 100% of the pixels in the sweep range are tested. n/9.99 = % covered')
    parser.add_argument('--refresh', type=bool_type, default=True)
    parser.add_argument('--p', type=bool_type, default=False, help='True to parallelize python processes. Note that that execution is already parallelized with the rendering backends')
    parser.add_argument('--n_cpus', type=int, default=-1, help='Override the automated, memory-aware calculation for number of cpus to use in python parallelization. -1 means do not override')
    parser.add_argument('--spp', type=int, default=-1, help='Override the spp used in the xml files with custom spp for rendering')
    parser.add_argument('--batch_factor', type=int, default=50, help='run parallel jobs in batches of size batch_factor * ncpus scenes per batch, terminate after each batch')
    parser.add_argument('--continue_iter', type=bool_type, default=False, help='if the last run was not complete, continue where it was left off')

    parser.add_argument('--y_max', type=float, default=0.1)
    parser.add_argument('--y_min', type=float, default=-0.9)

    parser.add_argument('--action', type=str, default='generate', help='action to take: generate | map_corr')

    args = parser.parse_args()
    return args
