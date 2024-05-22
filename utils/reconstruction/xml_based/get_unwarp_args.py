import argparse
from config import output_data_base_path, input_data_base_path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Totem Mapping dataset_gen_helpers')

    # Data paths
    parser.add_argument('--xml_data_path', type=str, help='base path to xml data used for the mappings', default=input_data_base_path)
    parser.add_argument('--output_data_path', type=str, help='base path to exp folder (excluding exp folder)', default=output_data_base_path)
    parser.add_argument('--exp_folder', type=str, required=True, help='name of exp folder')
    parser.add_argument('--nn_folder', type=str, required=True, help='name of the nn folder whose data will be used. Same as run_extension in get_nn_args.py')

    # Settings
    parser.add_argument('--im', type=str, default='smiley_face', help='filename of image to unwarp on (excluding res)')
    parser.add_argument('--im_fname', type=str, required=True) #TODO
    parser.add_argument('--res', type=int, required=True, help='resolution used to obtain the dataset in consideration. Must be consistent with the res of the mappings dataset that the NN trained on')
    parser.add_argument('--n_out', type=int, required=True, help='number of datapoints to rely on NN predictions for unwarping')

    args = parser.parse_args()
    return args