import os
import random

import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics

from utils.manip.parse_args import parse_arguments
from utils.generation.dict_based.scene import RedDotScene
import mitsuba as mi
from config import mi_variant
mi.set_variant(mi_variant)
args = parse_arguments()

import skimage

from math import log10, sqrt
bbox = RedDotScene(args.res, args.res, 128, is_render_only_totem=False, totem=args.totem).totem_bbox_ic
bbox_xmin, bbox_ymin = bbox[0]
bbox_xmax, bbox_ymax = bbox[1]
def load_and_manip_images(folder_of_npys, manip_type, **kwargs):
    for f in os.listdir(folder_of_npys):
        if f.endswith('npy') and not f.startswith('colorpatch') and not f.startswith('no_tot'):
            name = f.strip('.npy')
            im = np.load(os.path.join(folder_of_npys, f))
            if manip_type == 'colorpatch':
                savename = f'{manip_type}{name[8:]}'
                if os.path.join(folder_of_npys, (savename+'.npy')) in os.listdir(folder_of_npys):
                    print(f"SKIPPING MANIPULATING {name}")
                    continue
                color = kwargs.get('color', None)
                loc = kwargs.get('loc', None)
                if color is None:
                    color = [1,1,1]
                if loc is None:
                    # Randomly make square
                    x_start = random.randint(1, args.res - 100)
                    y_start = random.randint(1, args.res - 100)
                    loc = [[x_start, y_start], [x_start+random.randint(100,200), y_start+random.randint(200,440)]] # xmin, ymin, xmax, ymax
                    xmin, ymin = loc[0]
                    xmax, ymax = loc[1]
                    out_of_totem_bbox = (bbox_xmin > xmax or bbox_xmax < xmin) or (bbox_ymin > ymax)
                    while not out_of_totem_bbox:
                        x_start = random.randint(1, args.res - 1)
                        y_start = random.randint(1, args.res - 1)
                        loc = [[x_start, y_start], [x_start + random.randint(100, 200),
                                                    y_start + random.randint(100, 200)]]  # xmin, ymin, xmax, ymax
                        xmin, ymin = loc[0]
                        xmax, ymax = loc[1]
                        out_of_totem_bbox = (bbox_xmin > xmax or bbox_xmax < xmin) or (bbox_ymin > ymax)
                print(f"for image {name}: TOTEM BBOX XMIN {bbox_xmin} XMAX {bbox_xmax} YMIN {bbox_ymin} YMAX {bbox_ymax}")
                print(f"FINAL XMIN {xmin} XMAX {xmax} YMIN {ymin} YMAX {ymax}")
                xmin, ymin = loc[0]
                xmax, ymax = loc[1]
                im[ymin:ymax, xmin:xmax] = color
                manip_mask = np.zeros((args.res, args.res), dtype=np.uint8)
                manip_mask[ymin:ymax, xmin:xmax] = 1
                plt.imshow(im ** 0.5); plt.savefig(os.path.join(folder_of_npys, (savename+'.png')))
                np.save(os.path.join(folder_of_npys, (savename+'.npy')), im)

                #Save manip mask
                plt.imshow(manip_mask ** 0.5); plt.savefig(os.path.join(folder_of_npys, (savename+'_manipmask.png')))
                np.save(os.path.join(folder_of_npys, (savename+'_manipmask.npy')), manip_mask)

folder_of_npys = os.path.join(args.output_data_base_path, args.exp_folder, 'manipulations-3.0')

load_and_manip_images(folder_of_npys, 'colorpatch', color=[1,0,1])