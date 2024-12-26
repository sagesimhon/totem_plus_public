from __future__ import annotations      # so that we can use DotScene tying hints within the DotScene class

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np
from matplotlib import pyplot as plt

import mitsuba as mi
from config import mi_variant
mi.set_variant(mi_variant)

# define these defaults here to be used to generate transformations, as they are not stored anywhere in the scene dict.

# for camera transformations
default_camera_lookat = {
   'origin': (0, 0, 4),
   'target': (0, 0, 0),
   'up': (0, 1, 0)
}

default_camera_scale = (1, 1, 1)

# for red dot transformations
default_red_dot_depth = -3.0
default_world_range_x = [-2.0, 2.0]
default_world_range_y = [-2.0, 2.0]

default_red_dot_translate = (-0.6, -0.55, default_red_dot_depth)   # (-0.6, -0.55, -0.3)
default_red_dot_scale = (.03, .03, .03)   # (.02, .02, .02)     # make the dot small to capture better centroid resolution under warping
default_red_dot_color = [1, 0, 0]
red_dot_pixel_factor_estimate = 6 / 256
default_totem_center = (0.0, -0.545, 1.0)

# the below is used to deepcopy the template, dealing with the special cases where some values are of type
# mi.transformer (which cannot be cloned/serialized using regular deepcopy)
def deepcopy_dict_and_lists(o):
    if type(o) == list().__class__:
        return [deepcopy_dict_and_lists(v) for v in o]
    elif type(o) == dict().__class__:
        return {k: deepcopy_dict_and_lists(v) for k, v in o.items()}
    elif type(o) == tuple().__class__:
        return tuple(deepcopy_dict_and_lists(v) for v in o)
    else:
        return o


@dataclass
class RedDotScene:
    resolution_x: int
    resolution_y: int
    spp: int = 128
    red_dot_wc: tuple[float, float, float] = default_red_dot_translate
    red_dot_scale: tuple[float, float, float] = default_red_dot_scale
    name: str = 'base'
    is_render_only_totem: bool = True
    totem: str = 'sphere'
    totem_center: tuple[float, float, float] = default_totem_center
    scene_definition: dict = field(init=False)
    totem_bbox_wc: tuple[tuple[int, int], tuple[int, int]] = field(init=False)
    totem_bbox_ic: tuple[tuple[int, int], tuple[int, int]] = field(init=False)
    logger: logging.Logger = field(init=False)

    scene_template: ClassVar[dict] = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 5 # Default, may be changed in post init depending on self.totem
        },
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 5,
            'fov': 32,
            'sampler': {
                'type': 'independent',
                'sample_count': 128
            },
            'film': {
                'type': 'hdrfilm',
                'width': 256,
                'height': 256,
                'rfilter': {
                    'type': 'tent'  # 'gaussian'
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # BSDFs
        'grey': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.85, 0.85, 0.85]}},
        'white': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.885809, 0.698859, 0.666422]}},
        'green': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.105421, 0.37798, 0.076425]}},
        'brown': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.205, 0.05, 0.05]}},
        'red': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.570068, 0.0430135, 0.0443706]}},
        'black': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}},
        'glass': {'type': 'dielectric'},
        'mirror': {'type': 'conductor'},

        # Shapes
        'red_dot': {
            'type': 'obj',
            'filename': 'meshes/sphere.obj',
            'bsdf': {'type': 'ref', 'id': 'red'},
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': default_red_dot_color}
            }
        }


    }

    def _generate_totem_config(self):
        """
        Generate the configuration for the totem based on its type and transformations.

        Returns:
            dict: The configuration dictionary for the totem.
        """

        if self.totem == 'sphere':
            return {
                'type': 'sphere',
                'radius': 0.3,
                'center': self.totem_center,
                'bsdf': {'type': 'ref', 'id': 'glass'}
            }
        elif self.totem == 'sqp':
            return {
                'type': 'obj',
                'filename': 'meshes/square_pyramid.obj',
                'to_world': mi.ScalarTransform4f.translate((0.0, -0.15, 3.0)).scale((0.18, 0.18, 0.18)).rotate([1, 0, 0], 0),
                'bsdf': {'type': 'ref', 'id': 'glass'}
            }
        elif self.totem == 'teapot':
            return {
                'type': 'ply',
                'filename': 'meshes/teapot.ply',
                'to_world': mi.ScalarTransform4f.translate((0.0, -0.12, 3.0)).scale((0.05, 0.05, 0.05)).rotate([1, 0, 0], 90),
                'bsdf': {'type': 'ref', 'id': 'glass'}
            }
        elif self.totem == 'knot':
            return {
                'type': 'obj',
                'filename': 'meshes/Knot.obj',
                'to_world': mi.ScalarTransform4f.translate((0.0, -0.8, 0.0)).scale((0.05, 0.05, 0.05)).rotate([1, 0, 0], 0),
                'bsdf': {'type': 'ref', 'id': 'glass'}
            }
        elif self.totem == 'heart':
            return {
                'type': 'obj',
                'filename': 'meshes/love_heart.obj',
                'to_world': mi.ScalarTransform4f.translate((-0.35, -0.15, 0.0)).scale((0.3, 0.3, 0.3)).rotate([1, 0, 0], 0),
                'bsdf': {'type': 'ref', 'id': 'glass'}
            }

    def __post_init__(self):

        self.scene_template['totem'] = self._generate_totem_config()

        if self.totem == 'knot':
            self.scene_template['integrator']['max_depth'] = 8

        self.logger = logging.getLogger(self.__class__.__name__)
        self.scene_definition = deepcopy_dict_and_lists(RedDotScene.scene_template)

        camera = self.scene_definition['sensor']
        camera['film']['width'] = self.resolution_x
        camera['film']['height'] = self.resolution_y
        if self.spp > 0:
            camera['sampler']['sample_count'] = self.spp

        self.__compute_totem_bbox()
        if self.is_render_only_totem:
            film = self.scene_definition['sensor']['film']
            film['crop_offset_x'] = self.totem_bbox_ic[0][0]
            film['crop_offset_y'] = self.totem_bbox_ic[0][1]
            film['crop_width'] = abs(self.totem_bbox_ic[1][0] - self.totem_bbox_ic[0][0])
            film['crop_height'] = abs(self.totem_bbox_ic[1][1] - self.totem_bbox_ic[0][1] + 2)

    # mitsuba transforms are not serializable, so we create a method to set them manually -- must call it accordingly
    # as it would otherwise be called in the post_init constructor
    def set_transforms(self):
        # camera pose
        camera = self.scene_definition['sensor']
        camera['to_world'] = mi.ScalarTransform4f.look_at(
                origin=default_camera_lookat['origin'],
                target=default_camera_lookat['target'],
                up=default_camera_lookat['up']
            ) @ mi.ScalarTransform4f.scale(default_camera_scale)
        # red dot pose
        red_dot = self.scene_definition['red_dot']
        red_dot['to_world'] = mi.ScalarTransform4f.translate(self.red_dot_wc).scale(self.red_dot_scale)

    def clone(self,
              resolution_x: int = None,
              resolution_y: int = None,
              spp: int = None,
              red_dot_wc: tuple[float, float, float] = None,
              name: str = None) -> RedDotScene:
        """
           Create a new instance of `RedDotScene` by cloning the current instance,
           with the option to override specific attributes.

           Args:
               resolution_x (int, optional): The horizontal resolution for the new instance. Defaults to the current instance's `resolution_x`.
               resolution_y (int, optional): The vertical resolution for the new instance. Defaults to the current instance's `resolution_y`.
               spp (int, optional): Samples per pixel for the new instance. Defaults to the current instance's `spp`.
               red_dot_wc (tuple[float, float, float], optional): The world coordinates of the red dot for the new instance. Defaults to the current instance's `red_dot_wc`.
               name (str, optional): The name of the new instance. Defaults to the current instance's `name`.

           Returns:
               RedDotScene: A new `RedDotScene` instance with the specified or inherited attributes.
        """

        return RedDotScene(
            resolution_x=resolution_x or self.resolution_x,
            resolution_y=resolution_y or self.resolution_y,
            spp=spp or self.spp,
            red_dot_scale=self.red_dot_scale,
            red_dot_wc=red_dot_wc if red_dot_wc is not None else self.red_dot_wc,
            name=name or self.name,
            is_render_only_totem=self.is_render_only_totem,
            totem=self.totem)

    def __compute_totem_bbox(self):
        """
            Compute and set the bounding box (bbox) of the totem in image coordinates (IC)
            based on the totem type and resolution.

            Notes:
                - For 'sphere', the bounding box is computed dynamically using `world_to_image` to transform
                  world coordinates into image coordinates.
                - For other totems, bounding boxes are hardcoded for specific resolutions.
        """

        if self.totem == 'sqp':
            # CURRENTLY HARDCODED FOR SQUARE PYRAMID
            if self.resolution_x == 256:
                self.totem_bbox_ic = ((75, 110),(175,205))
            else:
                raise NotImplementedError(
                    f"NEED TO HARDCODE IN TOTEM BBOX FOR {self.totem} AT RES {self.resolution_x}!!!:)")

        elif self.totem == 'teapot':
            # CURRENTLY HARDCODED FOR TEAPOT
            if self.resolution_x == 256:
                self.totem_bbox_ic = ((55, 180), (205, 250))
            elif self.resolution_x == 1024:
                self.totem_bbox_ic = ((220, 725), (805, 1020))
            elif self.resolution_x == 512:
                self.totem_bbox_ic = ((120, 354), (404, 509))
            else:
                raise NotImplementedError(
                    f"NEED TO HARDCODE IN TOTEM BBOX FOR {self.totem} AT RES {self.resolution_x}!!!:)")

        elif self.totem == 'knot':
            # CURRENTLY HARDCODED FOR TOROID KNOT
            if self.resolution_x == 256:
                self.totem_bbox_ic = ((50, 177), (205, 253))
            elif self.resolution_x == 1024:
                self.totem_bbox_ic = ((209, 720), (829, 1020))
            elif self.resolution_x == 2048:
                self.totem_bbox_ic = ((415, 1433), (1653, 2044))
            else:
                raise NotImplementedError(
                    f"NEED TO HARDCODE IN TOTEM BBOX FOR {self.totem} AT RES {self.resolution_x}!!!:)")

        elif self.totem == 'heart':
            # CURRENTLY HARDCODED FOR HEART
            if self.resolution_x == 256:
                self.totem_bbox_ic = ((40, 64), (137, 148))
            elif self.resolution_x == 1024:
                self.totem_bbox_ic = ((164, 260), (549, 595))
            elif self.resolution_x == 2048:
                self.totem_bbox_ic = ((334, 520), (1098, 1188))
            elif self.resolution_x == 512:
                self.totem_bbox_ic = ((82, 130), (273, 293))
            else:
                raise NotImplementedError("Heart bbox not implemented for this res")

        elif self.totem == 'sphere':
            center_wc = np.array(self.scene_definition['totem']['center'])
            radius_wc = self.scene_definition['totem']['radius']
            min_wc = np.array([center_wc[0] - radius_wc, center_wc[1] - radius_wc, center_wc[2]])
            max_wc = np.array([center_wc[0] + radius_wc, center_wc[1] + radius_wc, center_wc[2]])
            self.totem_bbox_wc = min_wc, max_wc
            # flip_y_convention = True; do the following:
            min_ic = self.world_to_image(min_wc, flip_y_convention=True)
            max_ic = self.world_to_image(max_wc, flip_y_convention=True)
            xmin, ymax = min_ic
            xmax, ymin = max_ic
            self.totem_bbox_ic = ((xmin, ymin), (xmax, ymax))

    def world_to_image(self, world_coord, flip_y_convention=False) -> (int, int):
        """
            Convert a 3D world coordinate to 2D image coordinates using perspective projection.

            Args:
                world_coord (numpy.ndarray): The (x, y, z) coordinate in world space.
                flip_y_convention (bool): If True, flips the y-axis to match top-left origin systems. Defaults to False.

            Returns:
                tuple[int, int]: The (x, y) image coordinates corresponding to the world coordinate.
        """

        # Todo: this might be replaceable by mi.ScalarTransform4f.look_at
        camera = self.scene_definition['sensor']
        fov = camera['fov']
        near = camera['near_clip']
        far = camera['far_clip']
        origin = np.array(default_camera_lookat['origin'])
        target = np.array(default_camera_lookat['target'])
        up = np.array(default_camera_lookat['up'])

        # Compute the view matrix
        forward = target - origin
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        view_matrix = np.eye(4)
        view_matrix[0, :-1] = right
        view_matrix[1, :-1] = up
        view_matrix[2, :-1] = -forward
        view_matrix[0, 3] = -np.dot(right, origin)
        view_matrix[1, 3] = -np.dot(up, origin)
        view_matrix[2, 3] = np.dot(forward, origin)

        # Compute the projection matrix
        aspect_ratio = self.resolution_x / self.resolution_y
        fov_rad = np.deg2rad(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = f / aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far + near) / (near - far)
        projection_matrix[3, 2] = -1.0
        projection_matrix[2, 3] = (2.0 * far * near) / (near - far)

        # Compute the transformation matrix
        world_coord = np.append(world_coord, 1)
        trans_matrix = np.dot(projection_matrix, np.dot(view_matrix, world_coord))

        # Convert to image coordinates
        x = ((trans_matrix[0] / trans_matrix[3]) + 1) * self.resolution_x / 2
        y = ((trans_matrix[1] / trans_matrix[3]) + 1) * self.resolution_y / 2

        if flip_y_convention:
            # Account for y axis being flipped in Matplotlib coordinate system and use this coordinate convention
            y = self.resolution_y - y

        return int(x + .05), int(y + .05)  # todo -- verify it is okay to quantize here

    def get_world_corners_of_a_rectangle(self, key='bg_img'):
        # TODO: TEST(WIP)
        warnings.warn("get_world_corners_of_a_rectangle has not been fully tested!", UserWarning)

        # Extract the 'to_world' transform from the scene definition
        to_world_transform = self.scene_definition[key]['to_world']

        # Define the rectangle's corners in object space (ranging from -1 to 1)
        rectangle_corners_object = [
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0]
        ]

        # Transform rectangle corners to world coordinates
        rectangle_corners_world = [to_world_transform.transform_affine(p) for p in rectangle_corners_object]
        return rectangle_corners_world

    def put_totem_in(self):
        """ Add the specified totem to the scene definition based on its type and transformation. """
        self.scene_definition['totem'] = self._generate_totem_config()

    def set_background_image(self, path_to_bg_image, depth=default_red_dot_depth):
        """
            Add a background image and a point light to the scene definition.

            Args:
                path_to_bg_image (str): Path to the background image file.
                depth (float, optional): The depth position of the background image. Defaults to -3.0.

            Returns:
                None: Updates the `scene_definition` attribute in place.
        """
        angle = 180 # independent of landscape/portrait

        self.scene_definition['bg_img'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], angle).translate([0, 0.065, depth]).scale([2.25, 2.25, 2.25]),
                # 'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], angle).translate([0, 0, depth]).scale([2, 2, 2]),
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'bitmap',
                        'filename': path_to_bg_image,
                    }
                }
            }

        self.scene_definition['light_1'] = {
        'type': 'point',
        'position': [0, 0, 0],
        'intensity': {
            'type': 'spectrum',
            'value': 27.0,
        }
    }

    def remove_totem_if_exists(self):
        """ Remove the totem from the scene definition if it exists. """

        if 'totem' in self.scene_definition:
            del self.scene_definition['totem']

    def modify_scene_for_postprocessing(self, bg_img=None, landscape=True, depth=default_red_dot_depth):
        """
            Modify the scene for postprocessing by removing the red dot and optionally adding a background image.

            Args:
                bg_img (str, optional): Path to the background image file. If None, no background is added. Defaults to None.
                landscape (bool, optional): Specifies whether the background image is in landscape orientation. Defaults to True.
                depth (float, optional): The depth position for the background image. Defaults to `default_red_dot_depth`.
        """

        if 'red_dot' in self.scene_definition:
            del self.scene_definition['red_dot']
        if bg_img is not None:
            self.set_background_image(bg_img, landscape=landscape, depth=depth)

    def shift_uv_tots(self, uv_tot, xmin, ymin):
        """
            Shift (u, v)_tot coordinates to the reference frame of the totem bounding box.

            Args:
                uv_tot (list[int, int]): The original (u, v) coordinates.
                xmin (int): The x-coordinate of the totem bounding box's origin.
                ymin (int): The y-coordinate of the totem bounding box's origin.

            Returns:
                list[int, int]: The shifted (u, v) coordinates.
        """
        x, y = uv_tot
        return [x - xmin, y - ymin]

    def in_bbox(self, x, y, buffer=0):
        """
            Check if a coordinate (x, y) is within the totem bounding box.
            Args:
                x (int): The x-coordinate to check.
                y (int): The y-coordinate to check.
                buffer (int, optional): An additional margin to expand the bounding box. Defaults to 0.

            Returns:
                bool: True if the coordinate is within the expanded bounding box, False otherwise.
        """
        xmin, ymin = self.totem_bbox_ic[0]
        xmin -= buffer
        ymin -= buffer
        xmax, ymax = self.totem_bbox_ic[1]
        xmax += buffer
        ymax += buffer
        return xmin <= x <= xmax and ymin <= y <= ymax

    def get_color_corr_or_space_covered(self, path_to_exp_folder, is_rev=True, custom_fname_suffix=None, vertical_gradient=False):
        # TODO CLEAN AND DOCUMENT
        # TODO update implementaiton for is_rev = False, use old code/fit it back in with changes to get this working

        bbox = self.totem_bbox_ic
        compiled_scene = mi.load_dict(self.scene_definition)
        self.logger.info(f'Rendering the basic scene for color correlation: {self.name}')
        image = mi.render(compiled_scene)
        img_cam = np.array(image)

        path_to_mappings = os.path.join(path_to_exp_folder, 'mappings.json')
        path_to_rev_mappings = os.path.join(path_to_exp_folder, 'mappings_cam_to_tot.json')

        if is_rev:
            path_to_use = path_to_rev_mappings
            savename_prefix = 'full_color_corr'  # since color corr assumes a 1:1 mapping, just use this to visualize the space
            # covered without a rainbow gradient
        else:
            path_to_use = path_to_mappings
            savename_prefix = 'color_corr_uniques'

        if custom_fname_suffix is not None:
            savename_suffix = '_' + custom_fname_suffix + '.png'
        else:
            savename_suffix = '.png'

        savename = savename_prefix + savename_suffix
        saveloc = os.path.join(path_to_exp_folder, savename)

        # Obtain the mappings. if is_rev = False, mappings are uv_tot -> uv_cam, else uv_cam -> uv_tot
        # Since many uv_cams are assigned to the same uv_tot, the reversed dictionary is much larger and has more data.
        with open(path_to_use, 'r') as f:
            pixel_map = json.load(f)

        # Define the bounding box coordinates of image_tot within image_cam
        x1, y1 = bbox[0]  # top left corner of the bounding box
        x1 = math.floor(x1)
        y1 = math.floor(y1)#int(1.05 * math.floor(y1))

        x2, y2 = bbox[1]  # bottom right corner of the bounding box
        x2 = math.ceil(x2) #int(0.95 * math.ceil(x2))
        y2 = math.ceil(y2)

        m = int((y2 - y1 + 1)) # use 0.65 since image in totem is much smaller than actual width/height of BBOX
        n = int((x2 - x1 + 1)) # use 0.65 since image in totem is much smaller than actual width/height of BBOX

        alpha = 0.7

        # Define the rainbow gradient
        if vertical_gradient:
            num_lvls = m * n
            gradient_cmap = plt.cm.get_cmap('jet', num_lvls)
            gradient_cmap.set_gamma(1.0)  # Increase the contrast of the color map
            gradient_grid = gradient_cmap(np.arange(0, num_lvls - 1))
            gradient_grid_halfalpha = gradient_grid.copy()
            gradient_grid_halfalpha[:, -1] = alpha

        alpha_channel = np.ones((self.resolution_y, self.resolution_x, 1), dtype=img_cam.dtype)
        img_cam_rgba = np.concatenate((img_cam, alpha_channel), axis=-1)
        img_cam_original = img_cam.copy()


        # CMAP, with changes across X and Y instead of just Y:
        def arr_creat(upperleft, upperright, lowerleft, lowerright):
            arr = np.linspace(np.linspace(lowerleft, lowerright, arrwidth),
                              np.linspace(upperleft, upperright, arrwidth), arrheight, dtype=int)
            return arr[:, :, None]

        # Make color region smaller than actual totem bbox due to image appearing in only a small region of the bbox
        if self.totem == 'sphere':
            min_x = int(.1 * n)#0
            max_x = int(0.90 * n)#n
            min_y = int(.28 * m)#0
            max_y = int(0.98 * m)#m
        elif self.totem == 'sqp':
            min_x = int(.33 * n)  # 0
            max_x = int(0.95 * n)  # n
            min_y = int(.1 * m)  # 0
            max_y = int(0.97 * m)  # m
        elif self.totem == 'teapot' or self.totem == 'knot' or self.totem == 'heart':
            min_x = 0  # 0
            max_x = int(n)  # n
            min_y = 0  # 0
            max_y = int(m)  # m
        arrwidth = max_x - min_x
        arrheight = max_y - min_y

        r = arr_creat(0, 255, 0, 255)
        g = arr_creat(0, 0, 255, 0)
        b = arr_creat(255, 255, 0, 0)

        img = np.concatenate([r, g, b], axis=2)
        img_scaled = img / 255.0

        plt.imshow(img_scaled)#, origin="lower")
        plt.show()

        special_new_merged_dict = {}
        # Iterate through the dictionary
        for uv_1, uv_2 in pixel_map.items():
            if is_rev:
                uv_tot = uv_2
                uv_cam = uv_1
            else:
                uv_tot = uv_1
                uv_cam = uv_2
            uv_cam_og = uv_cam
            uv_tot_og = uv_tot
            # Convert the coordinates from strings to tuples
            uv_tot = [int(num) for num in uv_tot.strip('()').split(',')]  # x,y
            uv_tot_in_bbox_frame = self.shift_uv_tots(uv_tot, x1, y2)
            uv_cam = [int(num) for num in uv_cam.strip('()').split(',')]  # x,y

            # if self.in_bbox(uv_cam[0], uv_cam[1], buffer=35):
            #     special_new_merged_dict[uv_cam_og] = uv_tot_og
            # else:
            #     continue
            if not vertical_gradient:
                idx_x = uv_tot_in_bbox_frame[0] - max_x
                idx_y = uv_tot_in_bbox_frame[1] - max_y
                # Fill in the scene/image pixel
                bg_pixel = img_cam_original[uv_cam[1], uv_cam[0]]
                # print(uv_tot_in_bbox_frame)
                # img_cam[uv_cam[1], uv_cam[0]] = bg_pixel * (1 - alpha) + img_scaled[idx[1] - m,idx[0] - n] * alpha
                img_cam[uv_cam[1], uv_cam[0]] = bg_pixel * (1 - alpha) + img_scaled[idx_y, idx_x] * alpha

                # Fill in the totem pixel
                bg_pixel_2 = img_cam_original[uv_tot[1], uv_tot[0]]
                # img_cam[uv_tot[1], uv_tot[0]] = bg_pixel_2 * (1 - alpha) + img_scaled[idx[1] - m,idx[0]- n] * alpha
                img_cam[uv_tot[1], uv_tot[0]] = bg_pixel_2 * (1 - alpha) + img_scaled[idx_y, idx_x] * alpha
            else:
                idx = uv_tot_in_bbox_frame[0] + uv_tot_in_bbox_frame[1] * n
                bg_pixel = img_cam_original[uv_cam[1], uv_cam[0]]
                img_cam[uv_cam[1], uv_cam[0]] = bg_pixel * (1 - alpha) + gradient_grid[idx, 0:3] * alpha
                bg_pixel_2 = img_cam_original[uv_tot[1], uv_tot[0]]
                img_cam[uv_tot[1], uv_tot[0]] = bg_pixel_2 * (1 - alpha) + gradient_grid[idx, 0:3] * alpha

        # Show the output image
        npy_saveloc = saveloc.split('.png')[0] + '.npy'
        np.save(npy_saveloc, img_cam)

        plt.imshow(img_cam ** (1.0 / 2.0))
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()
        # plt.close()
        print(f"Color correlation successfully saved to {path_to_exp_folder}.")

        # f = '/Users/sage/data-for-totems-new-shapes/heart_merged/mappings_cam_to_tot_in_bbox.json'
        # with open(f, "w") as file:
        #     json.dump(special_new_merged_dict, file)



