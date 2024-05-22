import argparse
from config import input_data_base_path, output_data_base_path
from distutils.util import strtobool

from utils.generation.dict_based.scene import default_world_range_y

#TODO refactor input_data_base_path to input_data_base_path, everywhere in codebase

def bool_type(x):
    return bool(strtobool(x))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Totem Mapping detect_helpers')

    # Data paths
    parser.add_argument('--input_data_base_path', type=str, default=input_data_base_path, help='base path to input data (required xml files for generation)')
    parser.add_argument('--output_data_base_path', type=str, default=output_data_base_path, help='path to output data, which holds all experiments. Can live anywhere, including outside the project directory.')
    parser.add_argument('--exp_folder', type=str, required=True, help='name you would like to give this experiment', default='MyExperiment')

    # Settings (sweep granularity, resolution, rendering setting, parallelization settings)
    parser.add_argument('--res', type=int, required=True, help='desired resolution for desired mapping generation. Supported options are 728, 256 at the moment')
    parser.add_argument('--spp', type=int, default=-1, help='Override the spp used in the xml files with custom spp for rendering')

    parser.add_argument('--og', type=str, default='none', help='path+fname of ground truth png to compare against, path relative to input_data_base_path')
    parser.add_argument('--manip', type=str, default='none', help='path+fname of manipulated png, path relative to input_data_base_path')

    parser.add_argument('--result', type=str, default='none', help='path+fname of unwarped scene (.npy) to evaluate, path relative to output_data_base_path')
    parser.add_argument('--result_folder', type=str, default='unwarpings', help='path to folder of unwarped scenes (.npy) to evaluate, path relative to output_data_base_path/exp_folder')

    parser.add_argument('--grid_size', default=50, type=int)
    parser.add_argument('--patch_size', default=50, type=int)

    parser.add_argument('--totem', type=str, default='sphere', help='other supported options are teapot and sqp')


    args = parser.parse_args()
    return args
