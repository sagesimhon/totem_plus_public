import os
from utils.nn import get_nn_args as get_args
import utils.nn.totem_unwarp_dataset as totem_unwarp_dataset
from torch.utils.data import random_split
import math
import torch
from utils.generation.dict_based.scene import RedDotScene

args = get_args.parse_arguments()

data_path = os.path.join(args.base_mappings_data_path, args.exp_folder)

# Make place to store things
if not os.path.exists(os.path.join(data_path, 'nn')):
    os.mkdir(os.path.join(data_path, 'nn'))

if not os.path.exists(os.path.join(data_path, 'nn', args.run_extension)):
    os.mkdir(os.path.join(data_path, 'nn', args.run_extension))
    os.mkdir(os.path.join(data_path, 'nn', args.run_extension, 'iterative_plots'))

# Initialize datasets
train_dataset, val_dataset, in_mins, in_maxs, out_mins, out_maxs = totem_unwarp_dataset.get_train_val_dataset(data_path, ratio=0.80, hardcoded=(args.hard_cams is not None),
                                                                                                              hardcoded_path_cams=args.hard_cams, hardcoded_path_tots=args.hard_tots,
                                                                                                              hardcoded_complement_cams=args.hard_cams_comp, hardcoded_complement_tots=args.hard_tots_comp, res=args.res, save=False)
val_dataset, test_dataset = random_split(val_dataset, [len(val_dataset)//2,math.ceil(len(val_dataset)/2)])
# import pdb; pdb.set_trace()
scene = RedDotScene(args.res, args.res)
scene.set_transforms()
totem_bbox = scene.totem_bbox_ic
new_tot_mins, new_tot_maxs, new_cam_mins, new_cam_maxs = [torch.Tensor([282,740]), torch.Tensor([794,975]), torch.Tensor([1,1]), torch.Tensor([1022,1022])] #todo un-hardcode for res /shape

if args.is_manual_testset:
    raise DeprecationWarning("WARNING: Haven't tested this setting with new args post hardcoding. Proceed with caution...")
    tv_inputs = torch.load(args.manual_testset_inputs) # .pt file type
    tv_outputs = torch.load(args.manual_testset_outputs) # .pt file type
    manual_val_dataset = totem_unwarp_dataset.TotemUnwarpDataset(tv_inputs, tv_outputs)
    other_val_set, other_test_dataset = random_split(manual_val_dataset, [len(manual_val_dataset) // 2, math.ceil(
        len(manual_val_dataset) / 2)])  # TODO update hardcoding for this
    test_dataset = other_test_dataset
