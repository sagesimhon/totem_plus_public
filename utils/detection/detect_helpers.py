import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics

from utils.detection.parse_args import parse_detect_arguments
import mitsuba as mi
from config import mi_variant
mi.set_variant(mi_variant)
args = parse_detect_arguments()

import skimage

from math import log10, sqrt

def load_scene_images_from_folder(base_scene, og, manip=None, without_totem=True):
    images = {}
    if og:
        base_scene.set_background_image(os.path.join(args.input_data_base_path, og))
        images['original'] = np.array((mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
        if without_totem:
            del base_scene.scene_definition['totem']
            images['original_minus_totem'] = np.array((mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
            base_scene.put_totem_in()
    if args.manip != 'none':
        base_scene.set_background_image(os.path.join(args.input_data_base_path, args.manip))
        images['manip'] = np.array((mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
        if without_totem:
            del base_scene.scene_definition['totem']
            images['manip_minus_totem'] = np.array(
                (mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
            base_scene.put_totem_in()
    recon = np.load(os.path.join(args.output_data_base_path, args.exp_folder, args.result), allow_pickle=True).astype('float32')
    images['recon'] = recon  # np array
    return images  # len 2-3 list of three np (res, res, 3) RGB arrays of renderings

def load_scene_images_via_args(base_scene, without_totem=True):
    images = {}
    if args.og:
        base_scene.set_background_image(os.path.join(args.input_data_base_path, args.og))
        images['original'] = np.array((mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
        if without_totem:
            del base_scene.scene_definition['totem']
            images['original_minus_totem'] = np.array((mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
            base_scene.put_totem_in()
    if args.manip != 'none':
        base_scene.set_background_image(os.path.join(args.input_data_base_path, args.manip))
        images['manip'] = np.array((mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
        if without_totem:
            del base_scene.scene_definition['totem']
            images['manip_minus_totem'] = np.array(
                (mi.render(mi.load_dict(base_scene.scene_definition))))  # drjit array --> np array
            base_scene.put_totem_in()
    recon = np.load(os.path.join(args.output_data_base_path, args.exp_folder, args.result), allow_pickle=True).astype('float32')
    images['recon'] = recon  # np array
    return images  # len 2-3 list of three np (res, res, 3) RGB arrays of renderings


def save_og_and_manip_images(images, path):
    # Recon is already saved in exp folder, this function is to save renderings of OG and Manip for comparison/visualization
    # image_name = os.path.basename(args.og).strip('png')
    # path = os.path.join(args.output_data_base_path, f'detection_{image_name}')
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print(f"Warning: directory {path} already exists!")
    if 'original' in images and args.og != 'none':
        plt.imshow(images['original'] ** 0.5)
        suffix = os.path.basename(args.og)#.strip('.png')
        plt.savefig(os.path.join(path, f'original_{suffix}'))
        plt.close()
    if 'manip' in images and args.manip != 'none':
        plt.imshow(images['manip'] ** 0.5)
        suffix = os.path.basename(args.manip)#.strip('.png')
        plt.savefig(os.path.join(path, f'manip_{suffix}'))
        plt.close()


def get_patches_numpy(imA, imB, y_valid, grid_size, patch_size, protect_mask):
    # For nonoverlapping patches, grid_size must <= resolution and grid_size = res / patch_size
    # Grid size = num of strides per dim. Ie resolution / stride
    protect_mask_valid = protect_mask[:y_valid]

    # Crop and rescale to [-1, 1]

    imA = imA[:y_valid, :, :].astype(
        float)  # / 255 * 2 - 1 mitsuba already uses HDR range (0 min, 1.something max), so no need to normalize
    imB = imB[:y_valid, :, :].astype(float)  # / 255 * 2 - 1  mitsuba already uses HDR range, so no need to normalize
    imA = imA * protect_mask_valid[..., None]
    imB = imB * protect_mask_valid[..., None]
    H, W, _ = imA.shape

    patchesA = np.empty((grid_size ** 2, patch_size, patch_size, 3))  # assumes image is a square
    patchesB = np.empty((grid_size ** 2, patch_size, patch_size, 3))  # assumes image is a square

    idx = 0
    for i in np.linspace(0, H - patch_size, grid_size).astype(int):
        for j in np.linspace(0, W - patch_size, grid_size).astype(int):
            patchesA[idx] = imA[i:i + patch_size, j:j + patch_size]
            patchesB[idx] = imB[i:i + patch_size, j:j + patch_size]
            idx += 1

    return patchesA, patchesB

# def PSNR(mse):
#     if(mse == 0):  # MSE is zero means no noise is present in the signal.
#                   # Therefore PSNR has no importance.
#         return 100
#     max_pixel = 1.0
#     psnr = 20 * log10(max_pixel / sqrt(mse))
#     return psnr

def run_metrics(image_patches, recon_patches, image, recon, protect_mask, grid_size, patch_size, y_valid, out_dir, metric_name="L1",
                manip_mask=None, is_manip=None, do_psnr=True, is_subfolder=False, v=''):

    if is_manip is None:
        is_manip = ""
    protect_mask_valid = protect_mask[:y_valid]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    num = out_dir.split('___')[-1] if is_subfolder else ''

    # Output paths
    heatmap_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_heatmap.npy')
    vis_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_heatmap_vis.png')
    vis_path_2 = os.path.join(out_dir, f'{is_manip}P{patch_size}_heatmap_vis_alternate.png')
    vis_path_3 = os.path.join(out_dir, f'{is_manip}P{patch_size}_heatmap_vis_alternate_gammacorrected.png')
    overlay_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_heatmap_overlay.png')
    metrics_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_metrics.npy')
    logs_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_psnr.txt')
    logs_map_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_map.txt')
    all_psnr_path = os.path.join(os.path.dirname(out_dir), f'{is_manip}P{patch_size}_psnr_stats.txt')
    recon_with_psnr_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_recon_psnr_{num}.png')

    l1_average_path = os.path.join(out_dir, f'{is_manip}P{patch_size}_L1.npy')
    # if os.path.exists(vis_path):
    #     print(f"SKIPPING {vis_path}")
    #     if manip_mask is not None:
    #         with open(logs_map_path, 'r') as f:
    #             map = float(f.read())
    #     else:
    #         map = 'NA'
    #     return map, all_psnr_path
    # Save heatmap
    if metric_name == 'L1':
        heatmap = np.abs(image_patches - recon_patches)
    elif metric_name == 'L2':
        heatmap = np.square(image_patches - recon_patches)  # TODO check works
    else:
        raise NotImplementedError
    heatmap = np.reshape(heatmap, (grid_size ** 2, -1))
    normalized_array = np.linalg.norm(heatmap, axis=1, keepdims=True)

    # Calculate the average of all the elements in the new array
    average = np.mean(normalized_array)
    with open(l1_average_path, "w") as file:
        file.write(str(average))
    print(f'L1 AVG FOR {out_dir}: {average}')
    heatmap = np.reshape(np.mean(heatmap, axis=-1), (grid_size, grid_size))  # assumes image is a square
    np.save(heatmap_path, heatmap)

    # Save color heatmap
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], y_valid), cv2.INTER_CUBIC)
    plt.imsave(vis_path, heatmap_resized, cmap='jet')

    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(vis_path_2, bbox_inches='tight')
    plt.close()

    plt.imshow(heatmap ** 0.5)
    plt.colorbar()
    plt.savefig(vis_path_3, bbox_inches='tight')
    plt.close()

    # Save overlay
    alpha = 0.3  # for colormap
    bn = 0.6  # for unprotected region
    heatmap_cmap = imageio.imread(vis_path)[..., :3]
    # image = image ** 0.5  # edit gamma correction
    top_im = image[:y_valid]
    top_im = cv2.cvtColor(cv2.cvtColor(top_im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # 3-channel grayscale
    top_im = np.clip(top_im * 255, 0, 255)  # edit to fix 0-"1" HDR
    top = top_im * (1 - protect_mask_valid[..., None]) * bn + (top_im * (1 - alpha) + heatmap_cmap * alpha) * \
          protect_mask_valid[..., None]

    overlay = image.copy()
    overlay = cv2.cvtColor(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # 3-channel grayscale #TODO this is 0,1 range whereas top is 0,255! works only when y_valid is res so we dont see any of overlay wihtout top
    overlay = overlay * bn
    overlay[:y_valid] = top
    overlay = overlay.astype('uint8')
    imageio.imwrite(overlay_path, overlay)
    print(f'Detection results saved in: {out_dir}')

    # Save metrics: PSNR
    # todo to ensure this is only utilizing protected region (not just y valid), image(s) will need to be cropped
    if do_psnr:
        psnr = skimage.metrics.peak_signal_noise_ratio(np.clip(image,0,1)[:1250], (np.clip(recon,0,1)[:1250])) # compare LDR normalized values instead of HDR
        print(f'PSNR: {psnr}')
        with open(logs_path, 'w') as f:
            f.write(str(psnr))
        with open(all_psnr_path, 'a') as f:
            f.write(str(psnr)+'\n')
        plt.imshow(recon ** 0.5)
        plt.suptitle(f"Epoch {num}")
        plt.title(f"PSNR: {round(psnr,2)}")
        plt.savefig(recon_with_psnr_path, bbox_inches='tight')

    # Save metrics: patch-wise mAP
    if manip_mask is not None:
        manip_gt = (manip_mask[:y_valid, :] == 1)  # edited for dims res x res
        assert np.sum(manip_gt) > 0, "Manipulation mask must not be blank"
        manip_est = heatmap_resized.copy()
        protect_filter = np.where(protect_mask_valid)

        # Only evaluate protected pixels
        gt = manip_gt[protect_filter]
        est = manip_est[protect_filter]
        mAP = metrics.average_precision_score(gt, est) #TODO CAVEAT manip mask loaded is the npy arr not ** 0.5!!!

        print(f'Patch-wise mAP {metric_name}: {mAP}')

        with open(logs_map_path, 'w') as f:
            f.write(str(mAP))

        out = dict()
        out[f'patch_mAP_{metric_name}'] = mAP
        np.save(metrics_path, out)
        return mAP, all_psnr_path

    else: return 'NA', all_psnr_path
