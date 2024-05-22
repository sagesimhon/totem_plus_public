import numpy as np
import matplotlib.pyplot as plt

from config import mi_variant
import json
import math
import os

import mitsuba as mi
mi.set_variant(mi_variant)

def shift_uv_tots(uv_tot, xmin, ymin, xmax, ymax, resx, resy):
    x, y = uv_tot
    ymin_for_crop = int(resy - ymax)
    xmin_for_crop = int(xmin)

    y_shifted = y - ymin_for_crop
    x_shifted = x - xmin_for_crop
    return [x_shifted, y_shifted]

def get_color_corr(fname, path_to_fname, path_to_exp_folder, is_rev=True, bg_img='duck'):
    bboxgetter = Coords(fname, path_to_fname, is_reddot=False)
    bbox = bboxgetter.bounding_box
    img_cam = np.array(mi.render(mi.load_file(os.path.join(path_to_fname,fname))))
    path_to_mappings = os.path.join(path_to_exp_folder, 'mappings.json')
    path_to_rev_mappings = os.path.join(path_to_exp_folder, 'mappings_cam_to_tot.json')

    if is_rev:
        path_to_mappings = path_to_rev_mappings
        savename_prefix = 'space_covered' # since color corr assumes a 1:1 mapping, just use this to visualize the space
        # covered without a rainbow gradient
    else:
        savename_prefix = 'color_corr'
    if bg_img:
        savename_suffix = '_'+bg_img+'.png'
    else:
        savename_suffix = '.png'

    savename = savename_prefix+savename_suffix
    saveloc = os.path.join(path_to_exp_folder, savename)

    # Obtain the mappings. if is_rev = False, mappings are uv_tot -> uv_cam, else uv_cam -> uv_tot
    # Since many uv_cams are assigned to the same uv_tot, the reversed dictionary is much larger and has more data.
    with open(path_to_mappings, 'r') as f:
        pixel_map = json.load(f)

    # Define the bounding box coordinates of image_tot within image_cam
    x1, y1 = bbox[0]  # top left corner of the bounding box
    x1 = math.floor(x1)
    y1 = math.floor(y1)

    x2, y2 = bbox[1]  # bottom right corner of the bounding box
    x2 = math.ceil(x2)
    y2 = math.ceil(y2)

    if not is_rev:
        m = y2 - y1 + 1
        n = x2 - x1 + 1
    else:
        m = int(bboxgetter.res_y)
        n = int(bboxgetter.res_x)

    # Define the rainbow gradient
    num_lvls= m*n
    gradient_cmap = plt.cm.get_cmap('rainbow', num_lvls)
    gradient_cmap.set_gamma(2.0)  # Increase the contrast of the color map
    gradient_grid = gradient_cmap(np.arange(0, num_lvls-1))
    gradient_grid_halfalpha = gradient_grid.copy()
    gradient_grid_halfalpha[:, -1] = 0.5

    # plt.imshow(img_cam ** (1.0/2.0))

    # Iterate through the dictionary
    for uv_1, uv_2 in pixel_map.items():
        if is_rev:
            uv_tot = uv_2
            uv_cam = uv_1
        else:
            uv_tot = uv_1
            uv_cam = uv_2
        # Convert the coordinates from strings to tuples
        uv_tot = [int(num) for num in uv_tot.strip('()').split(',')] #x,y
        uv_tot_in_bbox_frame = shift_uv_tots(uv_tot, x1, y1, x2, y2, bboxgetter.res_x, bboxgetter.res_y)
        uv_cam = [int(num) for num in uv_cam.strip('()').split(',')] #x,y
        # Calculate the corresponding color in the gradient
        frac_tot = (uv_tot_in_bbox_frame[0] + uv_tot_in_bbox_frame[1]*n)/(m*n)
        idx = uv_tot_in_bbox_frame[0] + uv_tot_in_bbox_frame[1]*n
        img_cam[uv_tot[1], uv_tot[0]] = gradient_grid[idx, 0:3]
        img_cam[uv_cam[1], uv_cam[0]] = gradient_grid[idx, 0:3]

    # Show the output image
    npy_saveloc = saveloc.split('.png')[0]+'.npy'
    np.save(npy_saveloc, img_cam)
    plt.imshow(img_cam ** (1.0/2.0))
    plt.savefig(saveloc)
    # plt.show()
    plt.close()
    print(f"Color correlation successfully saved to {path_to_exp_folder}.")
