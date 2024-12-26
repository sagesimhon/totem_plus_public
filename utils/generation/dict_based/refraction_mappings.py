import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import mitsuba as mi

from config import mi_variant
from utils.generation.dict_based.scene import RedDotScene

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label, center_of_mass
from scipy.spatial.distance import cdist

import math
import os


@dataclass
class RedDotSceneTotemRefractionMapper:
    base_output_path: str
    scene: RedDotScene
    centroid_distance_threshold_res_factor: int = 13 / 2048
    centroid_distance_to_cam_threshold: int = 5
    img_color_threshold: float = .1

    image: mi.TensorXf = field(init=False)
    uv_cam: list[float] = field(init=False)
    uv_tot: list[float] = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        mi.set_variant(mi_variant)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.uv_tot = []

    def render_and_compute_refraction_mappings(self, is_save_plot=False):
        """
        Compute and set uv_tot/uv_cam mapping for a red dot in a scene with a single totem.
        Scene rendered in the process.
        """

        self.logger.debug(f'loading scene: {self.scene.name}')
        compiled_scene = mi.load_dict(self.scene.scene_definition)   # todo: evaluate if parallel=False is too slow
        self.logger.debug(f'Rendering scene: {self.scene.name}')
        self.image = mi.render(compiled_scene)
        self.logger.debug(f'Computing centroids for scene: {self.scene.name}')
        self.uv_cam = self.scene.world_to_image(self.scene.red_dot_wc, flip_y_convention=True)
        self.get_centroids(is_save_plot, is_many=(self.scene.totem != 'sphere'))      # sets uv_tot. todo: now its either 1,2 or 2,1 instead of 2,2 - clarify

    def translate_totem_coordinates_by_film_crop_offset(self, uv_tot):
        """
        Adjusts u,v coordinates by the film's crop offsets from the scene definition.

        Args:
            uv_tot (tuple): Original image coordinates

        Returns:
            tuple: Translated image coordinates.
        """
        x = uv_tot[0] + self.scene.scene_definition['sensor']['film'].get('crop_offset_x', 0)  # x
        y = uv_tot[1] + self.scene.scene_definition['sensor']['film'].get('crop_offset_y', 0)  # y
        return x, y

    # def in_bbox(self, x, y):
    #     tot_bbox = self.scene.totem_bbox_ic
    #     xmin, ymin = tot_bbox[0]
    #     xmax, ymax = tot_bbox[1]
    #     return xmin <= x <= xmax and ymin <= y <= ymax

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def get_centroids(self, is_save_plot, is_many=False):
        """
        Identifies and filters centroids of connected non-black pixel clusters in the rendered image,
        then assigns relevant centroid to uv_tot attribute via an educated guess.

        Args:
            is_save_plot (bool): Whether to save a plot of the rendered results for debugging.
            is_many (bool, optional): Indicates if multiple centroids are expected. Defaults to False.

        Returns:
            None: Updates `self.uv_tot` with the calculated centroid(s).
        """
        # TODO test new implementation
        warnings.warn("The updated 'get_centroids' method hasn't been fully tested and may not work as expected.", UserWarning)

        if self.image is None:
            raise RuntimeError('No image rendered, cannot proceed.')

        # Convert drjit.ArrayBase to NumPy array
        self.logger.debug(f'Converting image (actual render): {self.scene.name}')
        np_img = np.array(self.image)

        # Identify centroids
        centroids, num_labels = self._find_centroids(np_img)
        filtered_centroids = self._filter_centroids_by_distance(centroids, num_labels)

        expected_centroids = 1 if self.scene.is_render_only_totem else 2
        if not is_many and len(filtered_centroids) > expected_centroids + 2:
            raise AssertionError(f"MORE THAN {expected_centroids + 2} filtered centroids, expected {expected_centroids}!"
                                 f" (u,v)_cam is {self.uv_cam}")

        filtered_centroids = self._refine_centroids(np_img, filtered_centroids, expected_centroids, is_many,
                                                    is_save_plot)

        # Assign the appropriate centroid(s)
        if self.scene.is_render_only_totem:
            uv_tot = self.flip_coords_and_translate_and_assert_valid(filtered_centroids[0])
        else:
            self.logger.debug(f'Labeling centroids for scene: {self.scene.name}')
            uv_cam_from_dot, uv_tot = self.flip_coords_and_label_centroids(filtered_centroids)
        self.uv_tot = uv_tot

        if is_save_plot:
            self.save_sweep_figures(np_img, filtered_centroids)

    def _find_centroids(self, np_img):
        """Finds centroids of connected non-black pixel clusters."""

        # Create a binary mask of non-black pixels, avoiding small
        # off-black pixels (not sure what their origin is)
        self.logger.debug(f'Filtering image for centroids: {self.scene.name}')
        mask = np.any(np_img > self.img_color_threshold, axis=2)
        labels, num_labels = label(mask)

        self.logger.debug(f'Calculating centroids: {self.scene.name} -- {num_labels} labels found')
        centroids = [
            center_of_mass(mask, labels, i) for i in range(1, num_labels + 1)  # Skip label 0 (background)
        ]
        return centroids, num_labels

    def _filter_centroids_by_distance(self, centroids, num_labels):
        """Filters centroids based on a distance threshold."""

        self.logger.debug(f'Filtering centroids: {self.scene.name} -- {num_labels}')
        filtered_centroids = []
        # TODO change to just consider all contiguous clusters?
        for i, c1 in enumerate(centroids):
            is_far_enough = all(
                cdist([c1], [c2]) > self.centroid_distance_threshold_res_factor * self.scene.resolution_x
                for j, c2 in enumerate(filtered_centroids) if j != i
            )
            if is_far_enough:
                filtered_centroids.append(c1)
        return [(int(round(c[0])), int(round(c[1]))) for c in filtered_centroids]

    def _refine_centroids(self, np_img, centroids, expected_count, is_many, is_save_plot):
        """Refines the centroid list to match the expected count."""

        sum_z = self._calculate_brightness(np_img, centroids)

        if len(centroids) < expected_count:
            self._handle_insufficient_centroids(np_img, centroids, is_save_plot, expected_count)
        elif len(centroids) > expected_count:
            # Remove possible false positives (i.e. camera blob masking as a totem blob)
            max_index = sum_z.index(max(sum_z))
            if self.euclidean_distance(
                    self.flip_coords_and_translate_and_assert_valid(centroids[max_index]),
                    self.uv_cam) < self.centroid_distance_to_cam_threshold:
                sum_z.pop(max_index)
                centroids.pop(max_index)

            centroids = self._remove_excess_centroids(centroids, sum_z, expected_count, is_save_plot)

        return centroids

    def _calculate_brightness(self, np_img, centroids):
        """Calculates the brightness of regions around each centroid, including boundary checks."""

        height, width = np_img.shape[:2]

        def get_pixel_sum(y, x):
            """Safely gets the pixel sum at (y, x) with boundary checks."""
            if 0 <= y < height and 0 <= x < width:
                return np.sum(np_img[y, x])
            return 0

        return [
            get_pixel_sum(y, x) +
            get_pixel_sum(y + 1, x) +
            get_pixel_sum(y, x + 1) +
            get_pixel_sum(y - 1, x) +
            get_pixel_sum(y, x - 1) +
            get_pixel_sum(y - 1, x - 1) +
            get_pixel_sum(y + 1, x - 1) +
            get_pixel_sum(y - 1, x + 1) +
            get_pixel_sum(y + 1, x + 1)
            for y, x in centroids
        ]

    def _get_neighbors(self, y, x, np_img):
        """Returns valid neighbor coordinates around a given point."""

        height, width = np_img.shape[:2]
        neighbors = [
            (y + dy, x + dx) for dy, dx in
            [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        ]
        return [(ny, nx) for ny, nx in neighbors if 0 <= ny < height and 0 <= nx < width]

    def _handle_insufficient_centroids(self, np_img, centroids, is_save_plot, expected_count):
        """Handles cases where fewer centroids are found than expected."""

        if is_save_plot:
            self.save_sweep_figures(np_img, centroids)
        raise AssertionError(f"Could not find enough centroids. Found: {len(centroids)}, expected {expected_count}.")

    def _remove_excess_centroids(self, centroids, sum_z, expected_count, is_save_plot):
        """Removes dimmest centroids until the expected count is reached."""

        while len(centroids) > expected_count:
            min_index = sum_z.index(min(sum_z))
            centroids.pop(min_index)
            sum_z.pop(min_index)
        return centroids

    def save_sweep_figures(self, np_img, filtered_centroids):
        """Save rendering with details of found centroids."""

        np_img_plotting = np.copy(np_img)
        h, w = np_img_plotting.shape[0], np_img_plotting.shape[1]
        alpha_channel = np.ones((h, w, 1), dtype=np_img_plotting.dtype)
        rgba_img = np.concatenate((np_img_plotting, alpha_channel), axis=-1)

        marker_x_len = int(self.scene.resolution_x / 256 * 3) + 1
        marker_y_len = int(self.scene.resolution_y / 256 * 3) + 1

        for i, c in enumerate(filtered_centroids):  # y, x = + to down , + to right
            y, x = c
            # redmask = np.where(rgba_img[y - 3:y + 3, x - 3:x + 3] != [0,0,0,1], [0,0,0,1], rgba_img[y - 3:y + 3, x - 3:x + 3])
            # rgba_img[y - 3:y + 3, x - 3:x + 3] = [0, i, 1, 1] #* redmask

            rgba_img[y - marker_y_len:y + marker_y_len, x - 1: x + 1] = [0, i, 1, 1]
            rgba_img[y - 1: y + 1, x - marker_x_len:x + marker_x_len] = [0, i, 1, 1]

        plt.imshow(rgba_img ** (1.0 / 2.2), interpolation='none')
        num_c = len(filtered_centroids)
        if self.uv_tot:
            plt.title(f'{num_c} centroids found excluding those popped\n uv_cam: {self.uv_cam}; uv_tot: {self.uv_tot}')
        else:
            plt.title(f'{num_c} centroids found excluding those popped\n uv_cam: {self.uv_cam}')
        path = os.path.join(self.base_output_path, 'renderings')
        os.makedirs(path, exist_ok=True)
        centroids_fname = os.path.join(path, f'render_{self.scene.name}_centroids.png')
        plt.savefig(centroids_fname)
        plt.close()
        self.logger.info('Plotting complete')

    def render_only(self, is_save_plot=False):
        """
        Renders the scene without computing centroids, optionally saving the output.
        """
        #TODO test new implementation
        warnings.warn("The updated 'render_only' method hasn't been fully tested and may not work as expected.", UserWarning)

        self.scene.set_background_image()
        self.logger.debug(f'Loading scene: {self.scene.name}')
        compiled_scene = mi.load_dict(self.scene.scene_definition)  # Evaluate if parallel=False is too slow
        self.logger.debug(f'Rendering scene: {self.scene.name}')
        self.image = mi.render(compiled_scene)

        if is_save_plot:
            self.save_sweep_figures(np.array(self.image), [])

    def flip_coords_and_translate_and_assert_valid(self, uv_tot):
        xmin, ymin = self.scene.totem_bbox_ic[0]
        xmax, ymax = self.scene.totem_bbox_ic[1]

        # Inversion from y,x to x,y
        uv_tot = uv_tot[1], uv_tot[0]
        # Translate
        uv_tot = self.translate_totem_coordinates_by_film_crop_offset(uv_tot)
        # Check that the transformed coordinates exist within the totem bbox
        x, y = uv_tot
        assert xmin <= x <= xmax and ymin <= y <= ymax, 'Coordinate not correctly a totem coordinate!'
        return uv_tot

    # this is only used when we render the entire image
    def flip_coords_and_label_centroids(self, centroids):
        if len(centroids) > 2:
            raise "More than two centroids"

        c0, c1 = centroids

        # IMPORTANT STEP: invert coordinates from y,x to x,y
        c0 = c0[1], c0[0]
        c1 = c1[1], c1[0]

        c0_lbl = 'Cam'
        c1_lbl = 'Cam'

        xmin, ymin = self.scene.totem_bbox_ic[0]
        xmax, ymax = self.scene.totem_bbox_ic[1]

        # Check c0
        x, y = c0
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            c0_lbl = 'Totem'

        # Check c1
        x, y = c1
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            c1_lbl = 'Totem'

        assert c1_lbl != c0_lbl, "c0 and c1 have same label!"
        if c0_lbl == 'Totem' and c1_lbl == 'Cam':
            return c1, c0
        elif c0_lbl == 'Cam' and c1_lbl == 'Totem':
            return c0, c1
        else:
            raise AssertionError("c0 and c1 have same label and first assertion isn't working!")

    def show_img(self, show_bbox=False, show_lights=False, save_plots=False):
        if self.image is None:
            raise f'No image rendered, cannot proceed.'

        plt.axis("off")
        plt.imshow(self.image ** (1.0 / 2.2))
        plt.title("FULL SCENE")
        if save_plots:
            plot_fname = f"render_{self.scene.name}_full-scene.png"
            name = os.path.join(self.base_output_path, 'renderings', plot_fname)
            plt.savefig(name)
            plt.close()
        # plt.show()

        if show_bbox:
            xmin, ymin = self.scene.totem_bbox_ic[0]
            xmax, ymax = self.scene.totem_bbox_ic[1]

            cut = np.array(self.image)

            cut = cut[ymin:ymax, xmin:xmax]
            plt.imshow(cut ** (1.0 / 2.2))
            plt.title("BOUNDING BOX AROUND TOTEM")
            if save_plots:
                plot_fname = f"render_{self.scene.name}_bbox" + ".png"
                plot_fname = os.path.join(self.base_output_path, 'renderings', plot_fname)
                plt.savefig(plot_fname)
                plt.close()
            # plt.show()

        if show_lights:
            scene = self.get_scene_with_lights()
            img_lights = mi.render(scene)
            plt.imshow(img_lights ** (1.0 / 2.2))
            plt.title("LIGHTS ON")
            if save_plots:
                plot_fname = f"render_{self.scene.name}_LO.png"
                plot_fname = os.path.join(self.base_output_path, 'renderings', plot_fname)
                plt.savefig(plot_fname)
                plt.close()
            # plt.show()

    def get_scene_with_lights(self) -> mi.Scene:
        # TODO
        lo_scene_dict = deepcopy(self.scene.scene_definition)
        # ...
        raise NotImplementedError
        return mi.load_dict(lo_scene_dict)

