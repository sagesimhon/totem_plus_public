import argparse
from distutils.util import strtobool

from config import output_data_base_path, input_data_base_path

def bool_type(x):
    return bool(strtobool(x))
def parse_arguments():
    parser = argparse.ArgumentParser(description='Totem Mapping dataset_gen_helpers')

    # Data paths
    parser.add_argument('--xml_data_path', type=str, help='base path to xml data used for the mappings', default=input_data_base_path)
    parser.add_argument('--output_data_path', type=str, help='base path to exp folder (excluding exp folder)', default=output_data_base_path)
    parser.add_argument('--exp_folder', type=str, required=True, help='name of exp folder')
    parser.add_argument('--nn_folder', type=str, required=True, help='name of the nn folder whose data will be used. Same as run_extension in get_nn_args.py')

    # Settings
    # parser.add_argument('--im_fname', type=str, required=False, help='(full path) filename of image to unwarp (must be png)') #TODO: update unwapring code, so that this can be used if you want to only unwarp a single im
    # parser.add_argument('--im_for_naming', type=str, default='') #TODO
    parser.add_argument('--im_folder') # TODO documentation
    parser.add_argument('--res', type=int, required=True, help='resolution used to obtain the dataset in consideration. Must be consistent with the res of the mappings dataset that the NN trained on')
    parser.add_argument('--percent_nn', type=float, default=0, help='fraction of bg image datapoints to rely on NN predictions for unwarping')
    parser.add_argument('--use_nn', type=bool_type, default=True, help='if true: use NN for missing points. if false: use random noise from image to unscramble that pixel')
    parser.add_argument('--depth', type=float, default=-3.0)
    parser.add_argument('--totem_offset_x', type=float, default=0.0)
    parser.add_argument('--totem_offset_y', type=float, default=0.0)
    parser.add_argument('--is_iterate', type=bool_type, default=False)
    parser.add_argument('--manip', type=bool_type, default=False)
    parser.add_argument('--savefolder', type=str, default='unwarpings')
    parser.add_argument('--totem', type=str, default='sphere', help='other supported options are teapot and sqp')

    args = parser.parse_args()
    return args