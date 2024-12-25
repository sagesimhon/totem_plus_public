import math

import matplotlib.pyplot as plt

from models.mlp import MLP

import torch
import numpy as np
import json
import mitsuba as mi
import random

from train import inference
from utils.generation.dict_based.scene import RedDotScene
from utils.nn.embedder import get_embedder

import os
from utils.reconstruction.dict_based.get_unwarp_args import parse_arguments

from config import mi_variant
from utils.reconstruction.dict_based.unwarp_helpers import get_img, manip_im, to_arr

mi.set_variant(mi_variant)

custom_dataset_sizes = False
args = parse_arguments()
json_path = os.path.join(args.output_data_path, args.exp_folder, 'mappings_cam_to_tot.json')
tots_path = os.path.join(args.output_data_path, args.exp_folder, 'mappings.json')
manip_im_parent_path = 'data/mickey_bird'

#### Open dataset ####
with open(json_path, 'r') as f:
    reversed_mappings = json.load(f)
print(str(len(reversed_mappings)) + " mappings in file provided.")

#### Set up scene for unwarping ####
scene = RedDotScene(args.res, args.res, spp=128, is_render_only_totem=False, totem=args.totem, totem_center=(0.0 - args.totem_offset_x, -0.545 + args.totem_offset_y, 1.0))
scene.set_transforms()

path_to_images = os.path.join(args.xml_data_path, args.im_folder)
first_image_file = os.listdir(path_to_images)[0]
if not first_image_file.endswith('.png'):
    raise AssertionError(f'all contents of {path_to_images} must be png!')

is_landscape = args.im_folder == 'landscapes'
scene.modify_scene_for_postprocessing(os.path.join(path_to_images, first_image_file), landscape=is_landscape, depth=args.depth)
min_x, max_x = 0, args.res
min_y, max_y = 0, args.res

# im_to_tot: a 1 x H x W x 2 array of [u,v] coordinates. The 0,i,j,:th entry is the totem coordinate u,v that should
# be used to render the pixel i, j in the camera view.
im_to_tot = np.full((1, args.res, args.res, 2), [None, None])  # N x Hout x Wout x 2
im_to_tots = []
nn_count = 0
rand_count = 0
total_count = 0
image_bounds_x = range(min_x, max_x)
image_bounds_y = range(min_y, max_y)

n_bg_pixels = args.res * args.res #len(image_bounds_y) * len(image_bounds_x)
n_out = int(args.percent_nn * n_bg_pixels)
indices_to_infer = set(random.sample(range(n_bg_pixels), n_out))
random_inference_indices = set(
    [y * args.res + x for i, y in enumerate(image_bounds_y) for j, x in enumerate(image_bounds_x) if
     i * j in indices_to_infer])

inputs_parallel = []
for_NE_inputs_parallel = []
for_NE_indices = []
indices = []

rnd_points = set()
while True:
    rand_x = random.randint(min_x, max_x - 1)
    rand_y = random.randint(min_y, max_y - 1)
    point = (rand_x, rand_y)
    if point not in rnd_points:
        rnd_points.add(point)
    if len(rnd_points) >= args.percent_nn * n_bg_pixels:   # todo: need pixels of image
        break

if custom_dataset_sizes:
    with open('random_unassigned_stuff/closed-unique_coords/unique_coords_2048.json', 'r') as f:
        unique_coords = json.load(f)

    from utils.generation.dict_based.scene import red_dot_pixel_factor_estimate
    to_plot = []
    unique_coords_minus_bbox_2048 = []
    count_in_tot_bbox = 0
    notfounds = []
    for coord in unique_coords:
        coord_as_list = [float(num) for num in coord.strip('()').split(',')]
        x, y = coord_as_list
        xmin, ymin = scene.totem_bbox_ic[0]
        xmax, ymax = scene.totem_bbox_ic[1]
        buff_x = red_dot_pixel_factor_estimate * args.res
        buff_y = red_dot_pixel_factor_estimate * args.res
        to_plot.append(coord_as_list)
        if xmin - buff_x <= x <= xmax + buff_x and ymin - buff_y <= y <= ymax + buff_y:
            count_in_tot_bbox += 1
            continue
        else:
            unique_coords_minus_bbox_2048.append(coord_as_list)
            if coord not in reversed_mappings:
                notfounds.append(coord_as_list)

    unique_coords_as_set = set(unique_coords)

    # missing_as_arr = np.array(missing)
    to_plot_as_np = np.array(to_plot)

def in_bbox(x, y, tot_bbox):
    xmin, ymin = tot_bbox[0]
    xmax, ymax = tot_bbox[1]
    return xmin <= x <= xmax and ymin <= y <= ymax

for y in range(im_to_tot.shape[1]):  # y = i
    for x in range(im_to_tot.shape[2]):  # x = j
        if x not in image_bounds_x or y not in image_bounds_y:
            im_to_tot[0, y, x, :] = [0, 0]
        else:
            key = f'({str(x)}, {str(y)})'  # str repr of tuple/coord x,y
            if key in reversed_mappings: #and (x, y) not in rnd_points:
                im_to_tot[0, y, x, :] = to_arr(reversed_mappings[key]) #TODO update to include  - args.totem_offset_x*args.res # uv_tot
                total_count += 1
            elif args.use_nn:
                inputs_parallel.append([x, y])
                indices.append([y, x])
                inputs = torch.tensor(np.array([[x, y]]))
                total_count += 1
                nn_count += 1
            else:
                im_to_tot[0, y, x, :] = [0,0]
                # RANDOM NOISE: # im_to_tot[0, y, x, :] = [random.randint(0, args.res-1), random.randint(0, args.res-1)]
                total_count += 1
                rand_count += 1
if args.use_nn:
    #### Loading the pretrained NN ####
    metadata_path = os.path.join(args.output_data_path, args.exp_folder, 'nn', args.nn_folder, 'metadata.json')

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if not metadata['is_embedded']:
        embed_fn, mlp_input_dim = torch.nn.Identity(), 2
    else:
        embed_fn, mlp_input_dim = get_embedder(n_freqs=metadata['n_freqs'])
    model = MLP(D=metadata['d'], W=metadata['w'], input_ch=metadata['input_ch'],
                output_ch=metadata['output_ch'])  # , skips=[1000])
    state_dicts = []
    if args.is_iterate:
        for i in range(metadata['n_epochs']):#(352):#
            s = torch.load(os.path.join(args.output_data_path, args.exp_folder, 'nn', args.nn_folder, 'models',
                                       f'model_{i}.pt'))# f'model_e0_batch{i}.pt
            state_dicts.append(s)
    else:
        state_dicts = [
            torch.load(os.path.join(args.output_data_path, args.exp_folder, 'nn', args.nn_folder, 'model.pt'))]

    inputs_for_inference = torch.tensor(np.array(inputs_parallel))

    for i, state_dict in enumerate(state_dicts):
        print(f"On state dict {i}...")

        model.load_state_dict(state_dict)
        model.eval()
        uv_tots_from_inference = inference(model, inputs_for_inference, embed_fn, metadata['input_mins'],
                                           metadata['input_maxs'],
                                           metadata['output_mins'], metadata['output_maxs']).squeeze().numpy()
        uv_tots_from_inference[:, 0] = uv_tots_from_inference[:, 0] - math.ceil(args.totem_offset_x/1.55 *args.res)
        uv_tots_from_inference[:, 1] = uv_tots_from_inference[:, 1] - math.ceil(args.totem_offset_y/1.45 *args.res) #1.65 makes it lower than 1.55, 1.35 is too extreme

        indices = np.array(indices)
        im_to_tot_copy = np.copy(im_to_tot)
        im_to_tot_copy[0, indices[:, 0], indices[:, 1], :] = uv_tots_from_inference

        # normalize to the range required by 'input' of torch grid sample:
        # im_to_tot_copy = im_to_tot_copy.astype(np.int32)
        # im_to_tot_copy = im_to_tot_copy / 255.0  # Scale values between 0 and 1
        # im_to_tot_copy = im_to_tot_copy * 2.0 - 1.0  # Shift values between -1 and 1


        im_to_tots.append(im_to_tot_copy)

        print(f"Used NN {nn_count}/{total_count} times")
        print(f"Did random fill {rand_count}/{total_count} times")
        print("Done filling im_to_tot")

else:
    im_to_tots.append(im_to_tot)
    print("")

path_to_manips = os.path.join(args.output_data_path, args.exp_folder, f'realistic_manipulations{args.depth}_x_offset_{args.totem_offset_x}_y_offset_{args.totem_offset_y}', )

# Save each image unwarping
for f in os.listdir(path_to_images):
    if not f.endswith('.png'):
        continue
    else:
        print(f"Trying {f}...")
        name = f.strip('.png')
        image_folder = name + '_unwarped'
        path = os.path.join(args.output_data_path, args.exp_folder)
        if args.savefolder not in os.listdir(path):
            os.mkdir(os.path.join(path, args.savefolder))
        if image_folder in os.listdir(os.path.join(path, args.savefolder)) or (name+"_unwarped.png") in os.listdir(os.path.join(path, args.savefolder)):
            print(f"Skipping image {name}")
            continue
        else:
            if args.is_iterate:
                os.mkdir(os.path.join(path, args.savefolder, image_folder))
                path_for_saving = os.path.join(path, args.savefolder, image_folder)
            else:
                path_for_saving = os.path.join(path, args.savefolder)

            scene.modify_scene_for_postprocessing(os.path.join(path_to_images, f), landscape=is_landscape, depth=args.depth)
            im = get_img(scene.scene_definition)
            im = np.array(im)

            if not os.path.exists(path_to_manips):
                os.mkdir(path_to_manips)

            np.save(os.path.join(path_to_manips, 'original_' + name + '.npy'), im)
            plt.imshow(im ** 0.5)
            plt.savefig(os.path.join(path_to_manips, 'original_' + name + '.png'), bbox_inches='tight')

            # Now save without totem
            scene.remove_totem_if_exists()
            no_tot_im = get_img(scene.scene_definition)
            no_tot_im = np.array(no_tot_im)
            np.save(os.path.join(path_to_manips, 'no_tot_original_' + name + '.npy'), no_tot_im)
            plt.imshow(no_tot_im ** 0.5)
            plt.savefig(os.path.join(path_to_manips, 'no_tot_original_' + name + '.png'), bbox_inches='tight')

            if args.manip == True:
                colorpatch = False
                if colorpatch:
                    manip_im(im, path_to_manips, name, None)
                else:
                    manip_im_path = f'{manip_im_parent_path}/{name}.png'
                    if os.path.exists(manip_im_path):

                        scene.set_background_image(manip_im_path)

                        scene.remove_totem_if_exists()
                        manipped_rendered_im = mi.render(mi.load_dict(scene.scene_definition))
                        np.save(os.path.join(path_to_manips, f'no_tot_ps_{name}.npy'), manipped_rendered_im) # ps = 'photoshopped'
                        plt.imshow(manipped_rendered_im ** 0.5)
                        plt.savefig(os.path.join(path_to_manips, f'no_tot_ps_{name}.png'), bbox_inches='tight')

                        factor = 0.1
                        diff = manipped_rendered_im ** 0.5 - no_tot_im ** 0.5
                        mask = np.any(np.abs(diff) >= factor, axis=2).astype(np.uint8) * 1
                        np.save(os.path.join(path_to_manips, f'ps_{name}_manipmask.npy'), mask)
                        plt.imshow(mask)
                        plt.savefig(os.path.join(path_to_manips, f'ps_{name}_manipmask.png'), bbox_inches='tight')

            scene.modify_scene_for_postprocessing(os.path.join(path_to_images, f), landscape=is_landscape, depth=args.depth)
            scene.put_totem_in()
            im = get_img(scene.scene_definition)
            im = np.array(im)

            unwarped_im = np.full((args.res, args.res, 3), [None, None, None])

            for i, im_to_tot in enumerate(im_to_tots):
                for y, row in enumerate(im_to_tot[0, :, :, :]):
                    for x, col in enumerate(row):
                        coord = col[:]
                        try:
                            unwarped_im[y, x, :] = im[int(coord[1]), int(coord[0])]
                        except:
                            unwarped_im[y, x, :] = [0,0,0]
                            #raise AssertionError("out of bounds exception (from NN preds?)!!!")

                unwarped_im = unwarped_im.astype(np.float32)

                savename = os.path.join(path_for_saving, f'{i}_unwarped.png') if args.is_iterate \
                    else os.path.join(path_for_saving, name+'_unwarped.png')
                savename_np = os.path.join(path_for_saving, f'{i}_unwarped.npy') if args.is_iterate \
                    else os.path.join(path_for_saving, name+'_unwarped.npy')
                # savename = os.path.join(this_images_folder, f'{i}_unwarped.png')
                # savename_np = os.path.join(this_images_folder, f'{i}_unwarped.npy')

                savepath_np = os.path.join(path, args.savefolder, savename_np)
                np.save(savepath_np, unwarped_im)

                plt.imshow(unwarped_im ** (1.0 / 2.0))  # , interpolation='none')
                plt.savefig(os.path.join(path, args.savefolder, savename), bbox_inches='tight')
                # plt.show()
