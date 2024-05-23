from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
from utils.generation.dict_based.scene import RedDotScene, red_dot_pixel_factor_estimate, default_world_range_x, \
    default_world_range_y


@dataclass
class RedDotSceneSweeper:
    scene: RedDotScene
    sweep_density: float = 1
    sweep_range_x: list[float] = field(default_factory=lambda: default_world_range_x)
    sweep_range_y:  list[float] = field(default_factory=lambda: default_world_range_y)   # [-0.9, -0.1])
    sweep_range_z:  list[float] = field(default_factory=lambda: [-3.0, -3.0])
    scene_range_x: list[float] = field(default_factory=lambda: default_world_range_x)  # todo : from scene object
    scene_range_y: list[float] = field(default_factory=lambda: default_world_range_y)  # todo : from scene object
    scene_range_z: list[float] = field(default_factory=lambda: [-3, 4])      # todo : un-hardcode, get from scene object

    sweep_coords: Any = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sweep_coords = self.__generate_sweep_coords()      # populate in constructor, as it's used in logging

    def n_scenes(self) -> int:
        i = -1
        for i, scene in enumerate(self.scenes()):
            continue
        return i + 1

    def scenes(self, from_iter=0) -> Generator[RedDotScene, None, None]:
        self.sweep_coords = self.__generate_sweep_coords()      # todo: this is not needed here -- double check
        skipped = 0
        seen_coords = set()
        count_dupe_ics = 0
        seen_wcs = set()
        for i, coord in enumerate(self.sweep_coords):
            if i < from_iter:
                continue
            coords_str = f'{coord[0]:.3f},{coord[1]:.3f},{coord[2]:.3f}'
            scene = self.scene.clone(name=f'scene_{i}__pos_{coords_str}', red_dot_wc=coord)
            xmin, ymin = scene.totem_bbox_ic[0]
            xmax, ymax = scene.totem_bbox_ic[1]
            coord_ic = scene.world_to_image(coord, flip_y_convention=True)
            if (coord_ic[0], coord_ic[1]) in seen_coords:
                count_dupe_ics += 1
                #raise AssertionError("Image coord already tested for diff wc, this shouldn't happen!")
            else:
                seen_coords.add((coord_ic[0], coord_ic[1]))
            if (coord[0], coord[1]) in seen_wcs:
                raise AssertionError("World coord already tested, this shouldn't happen!")
            else:
                seen_wcs.add((coord[0], coord[1]))
            buff_x = red_dot_pixel_factor_estimate * self.scene.resolution_x
            buff_y = red_dot_pixel_factor_estimate * self.scene.resolution_y
            if xmin - buff_x <= coord_ic[0] <= xmax + buff_x and ymin - buff_y <= coord_ic[1] <= ymax + buff_y:
                skipped += 1
                continue
            yield scene

        ar = [f"({c[0]}, {c[1]})" for c in seen_coords]
        import json
        with open(f'unique_coords_{self.scene.resolution_x}_11-14.json', 'w') as f:
            json.dump(ar, f)
        self.logger.info(f'Skipped {skipped}')

    def __generate_sweep_coords(self) -> Any:
        min_x, max_x = self.sweep_range_x
        min_y, max_y = self.sweep_range_y
        min_z, max_z = self.sweep_range_z
        res_x = self.scene.resolution_x
        res_y = self.scene.resolution_y

        total_range_x = self.scene_range_x[1] - self.scene_range_x[0]
        total_range_y = self.scene_range_y[1] - self.scene_range_y[0]
        total_range_z = self.scene_range_z[1] - self.scene_range_z[0]

        # evenly spaced intervals
        # we add +1 because max_x/y/z is included in the bins, and when we split the ranges, we lose granularity
        x = np.linspace(min_x, max_x, num=math.ceil(self.sweep_density * res_x * (max_x - min_x) / total_range_x) + 1)
        y = np.linspace(min_y, max_y, num=math.ceil(self.sweep_density * res_y * (max_y - min_y) / total_range_y) + 1)
        z = np.linspace(min_z, max_z, num=math.ceil(self.sweep_density * max(res_x, res_y) * (max_z - min_z) / total_range_z) or 1)

        # create meshgrid of x, y, and z coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # flatten the meshgrid coordinates and add the initial coordinate to get final coordinates
        coords = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        self.logger.info(f'Coords shape: {coords.shape}')

        return coords

    def __generate_sweep_ranges(self) -> (list[float], list[float], list[float]):
        # todo: compute ranges automatically
        raise NotImplementedError('generate ranges')