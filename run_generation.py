import json
import logging
import os
import time

import mitsuba
from config import mi_variant

mitsuba.set_variant(mi_variant)    # must be set b4 imports, oddly not an issue on macos

from utils.generation.dict_based.get_mappings_args import parse_mappings_arguments
from utils.generation.dict_based.mappings_generator import MappingsGenerator
from utils.generation.dict_based.scene import RedDotScene

def generate_mappings(args):
    """
        Generate mappings dataset for rendering experiments.

        Args:
            args (Namespace): Command-line arguments containing configuration for mappings generation.

        Returns:
            None

        Saves output data in path_to_exp_name/
    """
    output_data_path = os.path.join(args.output_data_base_path, args.exp_folder)
    os.makedirs(output_data_path, exist_ok=True)
    from_iter = 0
    if args.continue_iter:
        f = os.path.join(output_data_path, 'progress.json')
        if os.path.exists(f):
            from_iter = json.load(open(f))['iter_num']
            if from_iter < 0:
                from_iter = 0
    if from_iter > 0:
        logging.info(f'Continuing from iteration: {from_iter}')

    t1 = time.time()
    mappings_generator = MappingsGenerator(
        output_data_path=output_data_path,
        resolution=args.res,
        spp=args.spp,
        sweep_density=args.sweep_density,
        is_parallel=args.p,
        n_cpus=args.n_cpus,
        batch_size_cups_factor=args.batch_factor,
        sweep_y=[args.y_min, args.y_max],
        sweep_x=[args.x_min, args.x_max],
        is_render_only_totem=args.is_render_only_totem,
        totem=args.totem
    )
    mappings_generator.run(from_iter=from_iter)
    t2 = time.time()
    logging.info(f'total runtime of mappings dataset generation: {t2 - t1}')


# Color correlation #
def obtain_color_corr(args, bg=None, fname_suffix=''):
    """
        Generate a color correlation visualization for the given scene.

        Args:
            args (Namespace): Command-line arguments containing configuration for color correlation.
            bg (str, optional): Path to the background image. Defaults to None.
            fname_suffix (str, optional): Custom filename suffix for output files. Defaults to ''.

        Returns:
            None

        Saves output data in path_to_exp_name/
    """
    output_data_path = os.path.join(args.output_data_base_path, args.exp_folder)
    scene = RedDotScene(args.res, args.res, args.spp, is_render_only_totem=False, totem=args.totem)
    scene.set_transforms()
    scene.modify_scene_for_postprocessing(bg)
    scene.get_color_corr_or_space_covered(output_data_path, is_rev=True, custom_fname_suffix=fname_suffix+'alpha0d7')


def main():
    """Parse arguments and execute the specified action (generate mappings or obtain color correlation)."""

    logging.basicConfig(format='%(asctime)s %(name)s.%(funcName)s [%(levelname)s]: %(message)s', level=logging.INFO)
    args = parse_mappings_arguments()
    if args.action == 'generate':
        generate_mappings(args)
    if args.action == 'map_corr':
        bg_img = os.path.join(args.input_data_base_path, 'image_to_unwarp', 'smiley_face.png')
        fname_suffix = ''
        obtain_color_corr(args, bg_img, fname_suffix)


if __name__ == "__main__":
    main()
