import xml.etree.ElementTree as ET
import os
from config import mi_variant
import mitsuba as mi
mi.set_variant(mi_variant)
import numpy as np
def to_arr(uv):
    # convert string representation in mapping to array of coordinate
    return [float(num) for num in uv.strip('()').split(',')]

def get_img(scene_dict):
    scene1 = mi.load_dict(scene_dict)
    image = mi.render(
        scene1)
    return image


def manip_im(im, savepath, name, box_loc):
    im[460:530, 345:405] = [0, 0, 0]
    savename = 'manipulated ' + name + '.npy'
    np.save(os.path.join(savepath, savename), im)
    # create + save manipulation mask
    manip_mask = np.zeros((728, 728))
    manip_mask[460:530, 345:405] = 1
    savename = 'manip_mask.npy'
    np.save(os.path.join(savepath, savename), manip_mask)
    # return im