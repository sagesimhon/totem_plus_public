from train import train, test, visualize_predictions, seed_worker#, g
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
import utils.nn.get_nn_args as get_args
import utils.nn.embedder as embedder
import utils.nn.totem_unwarp_dataset as totem_unwarp_dataset
import models.mlp as mlp
import random
from utils.nn.utils import get_writer
import json
import math

test_only = False
seed = 16
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)  # for potential sampler
# dataloader randomness handled in func with numpy random seed
# torch.cuda.manual_seed() when using GPU

# Get args
args = get_args.parse_arguments()

data_path = os.path.join(args.base_mappings_data_path, args.exp_folder)

# Make place to store things
if not os.path.exists(os.path.join(data_path, 'nn')):
    os.mkdir(os.path.join(data_path, 'nn'))

if not os.path.exists(os.path.join(data_path, 'nn', args.run_extension)):
    os.mkdir(os.path.join(data_path, 'nn', args.run_extension))
    os.mkdir(os.path.join(data_path, 'nn', args.run_extension, 'iterative_plots'))

if args.res_x == args.res_y:
    res_name = f'{args.res_x}'
else:
    res_name = f'{args.res_x}_x_{args.res_y}_y'

# Initialize datasets
train_dataset, val_dataset, in_mins, in_maxs, out_mins, out_maxs = totem_unwarp_dataset.get_train_val_dataset(data_path,
                                                                                                              ratio=0.80,
                                                                                                              hardcoded=(args.hard_cams is not None),
                                                                                                              hardcoded_path_cams=args.hard_cams,
                                                                                                              hardcoded_path_tots=args.hard_tots,
                                                                                                              hardcoded_complement_cams=args.hard_cams_comp,
                                                                                                              hardcoded_complement_tots=args.hard_tots_comp,
                                                                                                              res=res_name,
                                                                                                              save=False)
val_dataset, test_dataset = random_split(val_dataset, [len(val_dataset)//2,math.ceil(len(val_dataset)/2)]) # Split val set into val and test

#################################### IGNORE FOR NOW, HARDCODED FOR RES 1024 ####################################
do_manual = args.is_manual_testset
# tv_inputs = torch.load(f'NEW_{args.res}_testset_in.pt') # hard testset arg
# tv_outputs = torch.load(f'NEW_{args.res}_testset_out.pt') # hard testset arg

# new_tot_mins, new_tot_maxs, new_cam_mins, new_cam_maxs = [torch.Tensor([282,740]), torch.Tensor([794,975]), torch.Tensor([1,1]), torch.Tensor([1022,1022])] #todo un-hardcode for res /shape
# new_tot_mins, new_tot_maxs, new_cam_mins, new_cam_maxs = [torch.Tensor([282,740]), torch.Tensor([794,975]), torch.Tensor([1,1]), torch.Tensor([args.res-1,args.res-1])] #todo un-hardcode for res /shape

# manual_val_dataset = totem_unwarp_dataset.TotemUnwarpDataset(tv_inputs, tv_outputs)
# other_val_set, other_test_dataset = random_split(manual_val_dataset, [len(manual_val_dataset) // 2, math.ceil(
#     len(manual_val_dataset) / 2)])  # TODO update hardcoding for this

if do_manual:
    raise NotImplementedError("Current implementation hardcoded for specific experiment!")
    # test_dataset = other_test_dataset

################################################################################################################

# Initialize dataloaders
g = torch.Generator()
g.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, worker_init_fn=seed_worker,
                                               generator=g, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, worker_init_fn=seed_worker,
                                             generator=g, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), worker_init_fn=seed_worker,
                                             generator=g, shuffle=False)
# Initialize the embedder
if args.use_embed_fn:
    embed_fn, mlp_input_dim = embedder.get_embedder(n_freqs=args.n_freqs)
else:
    print("Not using embed fn!")
    embed_fn, mlp_input_dim = nn.Identity(), 2

# Initialize model
model = mlp.MLP(D=8, W=args.w, input_ch=mlp_input_dim, output_ch=2)
print("Created model.")
print(model)

# Criterion
criterion = nn.MSELoss()

# Optimizer
if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.4)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise AssertionError('Not implemented for your requested optimizer!')

# Training loop with a validation loop
# TODO: Add tensorboard logging (DONE)
# TODO: Add visualization/loader bar of training progress

writer = get_writer(model, None, data_path,
                    args.run_extension) # Tensorboard logging
# example_inputs, true_outputs = next(iter(train_dataloader)) # get the inputs in the 1st batch for the writer; data is a list of [inputs, labels]
# example_inputs = embed_fn(example_inputs)

# writer.add_graph(model, example_inputs)
saveloc = os.path.join(data_path, 'nn', args.run_extension)

if test_only:
    #### Loading the pretrained NN ####
    metadata_path = os.path.join(data_path, 'nn', args.run_extension, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if not metadata['is_embedded']:
        embed_fn, mlp_input_dim = torch.nn.Identity(), 2
    else:
        embed_fn, mlp_input_dim = embedder.get_embedder(n_freqs=metadata['n_freqs'])
    model = mlp.MLP(D=metadata['d'], W=metadata['w'], input_ch=metadata['input_ch'],
                output_ch=metadata['output_ch'])  # , skips=[1000])
    state_dict = torch.load(os.path.join(data_path, 'nn', args.run_extension, 'model.pt'))
    model.load_state_dict(state_dict)
    model.eval()
    final_epoch_loss = 00

else:
    final_epoch_loss = train(model, criterion, optimizer, writer, train_dataloader, val_dataloader, embed_fn, saveloc,
                             args.res_x, args.res_y,[in_mins, in_maxs, out_mins, out_maxs],
                             epochs=args.num_epochs, totem_name=args.totem)

testloader, test_error = test(model, criterion, test_dataloader, embed_fn, writer, in_mins, in_maxs, out_mins, out_maxs,
                              in_mins, in_maxs, out_mins, out_maxs, do_manual)

print(f"Test error {'manual' if do_manual else ''}", test_error)

visualize_predictions(model, in_mins, in_maxs, out_mins, out_maxs, in_mins, in_maxs, out_mins, out_maxs, embed_fn,
                      args.res_x, args.res_y, final_epoch_loss, test_error, train_dataloader, test_dataloader,
                      path_to_exp=data_path, ext=args.run_extension, is_embedded=args.use_embed_fn, lr=args.lr,
                      batch_size=args.batch_size, epochs=args.num_epochs, is_man=do_manual)

metadata = {'input_mins': in_mins.tolist(), 'input_maxs': in_maxs.tolist(), 'output_mins': out_mins.tolist(),
            'output_maxs': out_maxs.tolist(), 'd': args.d, 'w': args.w, 'input_ch': mlp_input_dim, 'output_ch': 2,
            'is_embedded': args.use_embed_fn, 'n_epochs': args.num_epochs, 'n_freqs': args.n_freqs}

with open(os.path.join(data_path, 'nn', args.run_extension, 'metadata.json'), "w") as json_file:
    json.dump(metadata, json_file, sort_keys=True, indent=4)

saveloc = os.path.join(data_path, 'nn', args.run_extension, "model.pt")
torch.save(model.state_dict(), saveloc)
print(f"Trained model, model metadata, and visualization of results successfully saved in {saveloc}!")
print("For viewing tensorboard visuals, in the terminal run the following:")
print(f"tensorboard --logdir={data_path}")

# in terminal, run:
# tensorboard --logdir=args.data_path (use literal value)