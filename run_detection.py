# Adapted from Jingweim/totems
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np

from utils.generation.dict_based.scene import RedDotScene
from utils.detection.detect_helpers import *

import mitsuba as mi
from config import mi_variant
mi.set_variant(mi_variant)
args = parse_detect_arguments()
run_none = False
run_manips = True
run_originals = False


def get_psnr_stats(path):
    # Report + save avg reconstruction PSNR
    total = 0
    all_psnrs = []
    count = 0
    with open(path, 'r') as file:
        for line in file:
            try:
                number = float(line.strip())
                total += number
                all_psnrs.append(number)
                count += 1
            except ValueError:
                # Handle cases where a line is not a valid number (e.g., ignore or log errors)
                print(f"Skipping invalid line: {line.strip()}")

    if count > 0:
        average = total / count
    else:
        print("No valid numbers found in the file.")

    data = np.array(all_psnrs)

    # Calculate the standard deviation using StandardScaler from scikit-learn
    scaler = StandardScaler()
    scaler.fit(data.reshape(-1, 1))
    std_deviation = scaler.scale_[0]

    print("Standard Deviation on psnr:", std_deviation)
    print(f'average psnr: {average}')

    path_to_exp = os.path.join(args.output_data_base_path, args.exp_folder)
    all_psnr_stats_for_this_exp_folder = os.path.join(path_to_exp, 'all_psnr_stats.txt')

    with open(all_psnr_stats_for_this_exp_folder, 'a') as f:
        if args.subfolder is not None:
            p = os.path.join(args.result_folder, args.subfolder)
        else:
            p = args.result_folder
        f.write(f"{p}: Avg {average}, STD {std_deviation}\n")
        f.flush()
        print(f"stats written to top level file at {all_psnr_stats_for_this_exp_folder}")

    # If using the NN for a percentage of the reconstruction (x axis), save to the psnr_v_percent-nn folder
    # if args.use_nn == True:
    #     try:
    #         arr = np.load(os.path.join(path_to_exp,'psnr_v_percent-nn.npy'))
    #         arr_updated = np.vstack((arr, np.array([args.x_axis, average])))
    #     except FileNotFoundError:
    #         arr_updated = np.array([args.x_axis, average])
    #
    #     np.save(os.path.join(path_to_exp,'psnr_v_percent-nn.npy'), arr_updated)
    #
    #     try:
    #         arr_2 = np.load(os.path.join(path_to_exp,'std_v_percent-nn.npy'))
    #         arr_updated = np.vstack((arr_2, np.array([args.x_axis, std_deviation])))
    #     except FileNotFoundError:
    #         arr_updated = np.array([args.x_axis, std_deviation])
    #     np.save(os.path.join(path_to_exp, 'std_v_percent-nn.npy'), arr_updated)
    #
    #     print(f"updated psnr arrays saved to {path_to_exp}: /std_v_percent-nn.npy and /psnr_v_percent-nn.npy")


def save_and_report_map(maps):
    # Report avg mAP
    if len(maps) > 0:
        print(f"AVG mAP: {sum(maps) / len(maps)}")
        data = np.array(maps)
        # Calculate the standard deviation using StandardScaler from scikit-learn
        scaler = StandardScaler()
        scaler.fit(data.reshape(-1, 1))
        std_deviation = scaler.scale_[0]
        print("mAP STD:", std_deviation)
    maps = np.array(maps)
    # TODO save map...

if __name__ == '__main__':

    if not run_none:
        base_scene = RedDotScene(args.res, args.res, 128, is_render_only_totem=False, totem=args.totem)
        base_scene.set_transforms()
        del base_scene.scene_definition['red_dot']

        # if args.dtype == 's':
        #     unwarped_im_name = os.path.basename(args.result).strip('.png')
        # elif args.dtype == 'm':

        path = os.path.join(args.output_data_base_path, args.exp_folder, args.result_folder)
        # manip_path = os.path.join(args.output_data_base_path, args.exp_folder, f'manipulations{args.depth}')
        prefix = 'ps'
        manip_path = os.path.join(args.output_data_base_path, args.exp_folder, 'realistic_manipulations-3.0_x_offset_0.0_y_offset_0.0')
        maps = []
        if args.subfolder is not None:
            path = os.path.join(path, args.subfolder)
            im_basename = args.subfolder.split('_unwarped')[0]
        for f in os.listdir(path):
            if f.endswith('.png'):
                if args.subfolder is None:
                    im_basename = f.strip('.png')[:-9]  # equivalent to desired effect of f.strip('_unwarped.png')
                    num = ''
                    detect_folder_path = os.path.join(args.output_data_base_path, args.exp_folder, args.result_folder, f'detection_{im_basename}')

                else:
                    num = f.split('_unwarped')[0]
                 #left off here... fix this and then fix run_metrics
                    detect_folder_path = os.path.join(args.output_data_base_path, args.exp_folder, args.result_folder, args.subfolder, f'detection_{im_basename}___{num}')

                # save_og_and_manip_images(images_as_dict, detect_folder_path)
                protect_mask = np.zeros((args.res, args.res))
                protect_mask[:,:] = 1
                # protect_mask[0:100,:] = 0
                if args.protect_totem:
                    xmin, ymin = base_scene.totem_bbox_ic[0]
                    xmax, ymax = base_scene.totem_bbox_ic[1]
                    protect_mask[ymin:ymax, xmin:xmax] = 0
                y_valid = args.res # TODO update

                recon = np.load(os.path.join(path, f.split('.png')[0]+'.npy')) ** 0.5#f'{im_basename}_unwarped.npy'))

                if run_originals:
                    original = np.load(os.path.join(manip_path, f'no_tot_original_{im_basename}.npy')) ** 0.5
                    patches_img, patches_recon = get_patches_numpy(original, recon, y_valid, args.grid_size, args.patch_size,
                                                                   protect_mask)
                    _, all_psnr_path = run_metrics(patches_img, patches_recon, original, recon, protect_mask,
                                args.grid_size, args.patch_size, y_valid, detect_folder_path, metric_name="L1", is_subfolder=args.subfolder)

                if run_manips:
                    #if os.path.exists(os.path.join(manip_path, f'colorpatch_{im_basename}.npy')): #TODO generalize prefix to all manip types
                    # im_basename_base = im_basename.split(' 2')[0]
                    if os.path.exists(os.path.join(manip_path, f'no_tot_{prefix}_{im_basename}.npy')):
                        # TODO: will save to same folder as recon-vs-og

                        manip_mask = np.load(os.path.join(manip_path, f'{prefix}_{im_basename}_manipmask.npy'))
                        # mani_mask = np.load(os.path.join(manip_path, 'BLACKMASK.npy'))
                        manip = np.load(os.path.join(manip_path, f'no_tot_{prefix}_{im_basename}.npy')) ** 0.5
                        patches_img, patches_recon = get_patches_numpy(manip, recon, y_valid, args.grid_size, args.patch_size, protect_mask)
                        map, _ = run_metrics(patches_img, patches_recon, manip, recon, protect_mask, args.grid_size, args.patch_size, y_valid, detect_folder_path, metric_name="L1", is_manip=f'{prefix}_', manip_mask=manip_mask, do_psnr=False)#manip_mask=manip_mask,
                        maps.append(map)

        if run_originals:
            get_psnr_stats(all_psnr_path)
        if run_manips:
            save_and_report_map(maps)

# TODO: update to gamma correction was only made for realistic manips. keep this consistent for future exps