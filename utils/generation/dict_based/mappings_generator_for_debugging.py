import logging
import math
import traceback

import psutil as psutil
import os
import json
import time
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from multiprocessing import get_context

from config import mi_variant
from utils.generation.dict_based.refraction_mappings import RedDotSceneTotemRefractionMapper
from utils.generation.dict_based.scene import RedDotScene
from utils.generation.dict_based.scene_sweeper import RedDotSceneSweeper


#
# for memory leak
# wr = weakref.ref(x)
# gc.get_referents(wr)
#


@dataclass
class MappingsGenerator:
    output_data_path: str = '.'
    resolution: int = 256
    spp: int = 128
    sweep_density: float = 1
    is_parallel: bool = False     # python parallelization to help manage memory leak, also provides slight improvement
    n_cpus: int = -1              # override memory aware free # cpu detection
    batch_size_cups_factor: int = 50
    sweep_y: list[float] = field(default_factory=lambda: [-0.9, 0.0])
    is_fail_silent: bool = False
    is_pool_terminating: bool = True
    parallel_method: str = 'fork'  # spawn
    is_render_only_totem: bool = True
    totem: str = 'sphere'

    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # compute the number of CPUs we can use based on the machine CPUs and the memory available
    def __get_cpus(self, n_spare_cpus=1, spare_memory_gb=0.5, memory_usage_required_per_process_gb=.4) -> int:
        if self.n_cpus > 0:
            return self.n_cpus
        n_cpus = os.cpu_count() - n_spare_cpus
        free_memory_gb = psutil.virtual_memory().available / 2**30
        max_cpus_from_memory = int((free_memory_gb - spare_memory_gb) / memory_usage_required_per_process_gb)
        if max_cpus_from_memory < 0:
            raise MemoryError(f'Not enough memory: {free_memory_gb}GB available, '
                              f'min required: {memory_usage_required_per_process_gb} ')
        if n_cpus > max_cpus_from_memory:
            print(f'adjusting cpus from {n_cpus} to {max_cpus_from_memory} due to memory constraints')
            n_cpus = max_cpus_from_memory
        # adjust available cpus for python parallelization if llvm is used
        factor = 3 if 'llvm' in mi_variant and n_cpus > 12 else 2 if 'llvm' in mi_variant else 1
        return math.ceil((n_cpus or 1) / factor)

    def __save_mappings(self, mappings: dict, reverse_mappings: dict, iter_times: list[float], iter_num: int):
        self.logger.info(f"Saving mappings to {self.output_data_path}")
        # mappings
        f = os.path.join(self.output_data_path, f'mappings.json')
        with open(f, "w") as file:
            json.dump(mappings, file)
        # reverse mappings
        f = os.path.join(self.output_data_path, f'mappings_cam_to_tot.json')
        with open(f, "w") as file:
            json.dump(reverse_mappings, file)
        # iter times
        f = os.path.join(self.output_data_path, 'iter_times.json')
        with open(f, "w") as file:
            json.dump(iter_times, file)
        # iter progress
        f = os.path.join(self.output_data_path, 'progress.json')
        with open(f, "w") as file:
            json.dump({'iter_num': iter_num}, file)

    def __load_mappings_if_continuing(self, iter_num: int) -> (dict, dict, list[float]):
        mappings = {}
        reverse_mappings = {}
        iter_times = []

        if iter_num == 0:
            return mappings, reverse_mappings, iter_times

        f = os.path.join(self.output_data_path, f'mappings.json')
        if os.path.exists(f):
            self.logger.info(f"Loading mappings from earlier runs in {self.output_data_path}")
            with open(f, 'r') as f:
                mappings = json.load(f)
        f = os.path.join(self.output_data_path, f'mappings_cam_to_tot.json')
        if os.path.exists(f):
            with open(f, 'r') as f:
                reverse_mappings = json.load(f)
        f = os.path.join(self.output_data_path, 'iter_times.json')
        if os.path.exists(f):
            with open(f, 'r') as f:
                iter_times = json.load(f)

        return mappings, reverse_mappings, iter_times

    def __plot_runtimes(self, iter_times: list[float]):
        plt.bar(range(len(iter_times)), iter_times)
        plt.title('Runtime for each iteration (each red dot file in sweep)')
        plt.xlabel('File index')
        plt.ylabel('Time (s)')
        plt.savefig(os.path.join(self.output_data_path, 'timing_iters.png'))
        plt.close()

    def run(self, from_iter=0):
        mappings, reverse_mappings, iter_times = self.__load_mappings_if_continuing(from_iter)
        self.logger.info('Preparing base red dot scene.')
        scene = RedDotScene(resolution_x=self.resolution, resolution_y=self.resolution, spp=self.spp, red_dot_scale=0.08, is_render_only_totem=False, totem=self.totem) #TODO remove non default scale and is-render-only totem after debugging 256
        scene_sweeper = RedDotSceneSweeper(scene, self.sweep_density, sweep_range_y=self.sweep_y)
        try:
            if self.is_parallel:
                mappings, reverse_mappings, iter_times, count_failed = self.run_parallel(scene_sweeper, from_iter)
            else:
                mappings, reverse_mappings, iter_times, count_failed = self.run_serial_just_render(scene_sweeper, from_iter) #TODO change back to just run_serial(...) after debugging
            self.__save_mappings(mappings, reverse_mappings, iter_times, -1)
            if count_failed:
                self.logger.warning(f'Failed count: {count_failed}')
            self.logger.info(f'{len(reverse_mappings)} total mappings found.')
            self.logger.info(f'{len(reverse_mappings) - len(mappings)} duplicate uv_tots for distinct uv_cams.')
            self.__plot_runtimes(iter_times)
        except Exception as e:              # KeyboardInterrupt:
            self.logger.error(e)
            traceback.print_exc()
            self.__plot_runtimes(iter_times)
            if not self.is_fail_silent:
                raise e
        return mappings, reverse_mappings

    def run_parallel(self, scene_sweeper: RedDotSceneSweeper, from_iter=0) -> (dict, dict, list[float], int):
        mappings, reverse_mappings, iter_times = self.__load_mappings_if_continuing(from_iter)
        fail_count = 0  # todo: save/load from file to continue counts

        self.logger.info(f'Generating scenes to distribute: {len(scene_sweeper.sweep_coords)} scenes')
        params = [[i, self.output_data_path, scene]
                  for i, scene in enumerate(scene_sweeper.scenes(from_iter), start=from_iter)]
        n_cpus = self.__get_cpus()
        self.logger.info(f'Distributing {len(params)} jobs across {n_cpus} parallel processes')

        # even with multiprocessing, stale references that cause memory leaks at each iteration remain.
        # Let's run in batches, and intermittently terminate all jobs to try to clear any memory leaks.
        # We must terminate (not close) and recreate a new pool. We do so every 100 (or so) cycles,
        # in order to avoid too much overhead in termination and recreation of the pool
        batch_size = n_cpus * self.batch_size_cups_factor
        param_batches = [params[i: i + batch_size] for i in range(0, len(params), batch_size)]
        start = time.time()
        for i, params in enumerate(param_batches):
            self.logger.info(f'Launching process pool with {n_cpus} processes for {len(params)} jobs.')
            with get_context(self.parallel_method).Pool(n_cpus) as pool:
                results = pool.starmap(MappingsGenerator.process_red_dot_scene, params)
                for result in results:
                    _mappings, _reverse_mappings, _iter_time, _is_error = result
                    mappings |= _mappings
                    reverse_mappings |= _reverse_mappings
                    iter_times.append(_iter_time)
                    if _is_error:
                        fail_count += 1
                self.__save_mappings(mappings, reverse_mappings, iter_times, max(param[0] for param in params))
                if self.is_pool_terminating:
                    self.logger.info(f'Terminating process pool {i}')
                    pool.terminate()   # try this to avoid memory leaks
                current_time = time.time()
                batch_time = (current_time - start) / (i + 1)
                file_time = batch_time / batch_size
                total_hr = batch_time * len(param_batches) / 60 / 60
                self.logger.info(
                    f'Progress: {i / len(param_batches) * 100:.1f}% (batch #{i}/{len(param_batches)}. '
                    f'Batch size={len(params)}), speeds=[{batch_time:.3f} sec/batch] [{file_time:.3f} sec/scene]. '
                    f'Total runtime -- {total_hr:.2f} hr')
        self.__save_mappings(mappings, reverse_mappings, iter_times, -1)
        return mappings, reverse_mappings, iter_times, fail_count

    def run_serial(self, scene_sweeper: RedDotSceneSweeper, from_iter=0) -> (dict, dict, list[float], int):
        mappings, reverse_mappings, iter_times = self.__load_mappings_if_continuing(from_iter)
        fail_count = 0  # todo: save/load from file to continue counts
        start = time.time()
        n_scenes = scene_sweeper.n_scenes()
        tot_x_err = 0
        tot_y_err = 0
        for i, scene in enumerate(scene_sweeper.scenes(from_iter), start=from_iter):
            result = self.process_red_dot_scene_only_render(i, self.output_data_path, scene)
            _mappings, _reverse_mappings, _iter_time, _is_error = result #, x_err, y_err = result
            #tot_x_err += x_err
            #tot_y_err += y_err
            mappings |= _mappings
            reverse_mappings |= _reverse_mappings
            iter_times.append(_iter_time)
            if _is_error:
                fail_count += 1
            if (i + 1) % self.batch_size_cups_factor == 0:
                batch_time = (time.time() - start) / (i - from_iter + 1)
                total_hr = batch_time * n_scenes / 60 / 60
                self.logger.info(f'Progress ({i / n_scenes * 100:.1f}%) {i} / {n_scenes} '
                                 f'--- {batch_time:.2f} sec/scene -- total {total_hr:.2f} hr') #\n tot_x_err: {tot_x_err} {tot_x_err/i}, tot_y_err: {tot_x_err} {tot_y_err/i}')
                self.__save_mappings(mappings, reverse_mappings, iter_times, i)
        self.__save_mappings(mappings, reverse_mappings, iter_times, -1)
        return mappings, reverse_mappings, iter_times, fail_count

    def run_serial_just_render(self, scene_sweeper: RedDotSceneSweeper, from_iter=0) -> (dict, dict, list[float], int):
        for i, scene in enumerate(scene_sweeper.scenes(from_iter), start=from_iter):
            self.process_red_dot_scene_only_render(i, self.output_data_path, scene)
        print("DONE!")

    @staticmethod
    def process_red_dot_scene(i: int, base_output_path: str, scene: RedDotScene) -> (dict, dict, float, bool):
        # when in parallel, we must reset the loging config
        logging.basicConfig(format='%(asctime)s %(name)s.%(funcName)s [%(levelname)s]: %(message)s', level=logging.INFO)
        mappings = {}
        reverse_mappings = {}
        is_error = False
        logging.info(f'Trying iter {i}, scene: {scene.name}')
        start_time = time.time()
        # errx, erry = 0, 0
        try:
            scene.set_transforms()
            refraction_mapper = RedDotSceneTotemRefractionMapper(base_output_path, scene)
            refraction_mapper.render_and_compute_refraction_mappings(is_save_plot=True)
            logging.debug(f'For scene {scene.name}, red dot world coords are {scene.red_dot_wc}')
            logging.debug(f'uv_cam, uv_tot: {refraction_mapper.uv_cam}, {refraction_mapper.uv_tot}')
            mappings[str(refraction_mapper.uv_tot)] = str(refraction_mapper.uv_cam)
            reverse_mappings[str(refraction_mapper.uv_cam)] = str(refraction_mapper.uv_tot)
            # errx, erry = abs(uv_cam[0] - refraction_mapper.uv_cam[0]), abs(uv_cam[1] - refraction_mapper.uv_cam[1])
        except AssertionError as e:
            logging.error(f'For scene {scene.name}, red dot world coords are {scene.red_dot_wc}\n {e}')
            is_error = True
        iter_time = time.time() - start_time
        return mappings, reverse_mappings, iter_time, is_error  # errx, erry

    @staticmethod
    def process_red_dot_scene_only_render(i: int, base_output_path: str, scene: RedDotScene) -> (dict, dict, float, bool):
        # when in parallel, we must reset the loging config
        logging.basicConfig(format='%(asctime)s %(name)s.%(funcName)s [%(levelname)s]: %(message)s', level=logging.INFO)
        mappings = {}
        reverse_mappings = {}
        is_error = False
        logging.info(f'Trying iter {i}, scene: {scene.name}')
        start_time = time.time()
        # errx, erry = 0, 0
        try:
            scene.set_transforms()
            refraction_mapper = RedDotSceneTotemRefractionMapper(base_output_path, scene)
            refraction_mapper.render_only()
            logging.debug(f'For scene {scene.name}, red dot world coords are {scene.red_dot_wc}')
        except AssertionError as e:
            logging.error(f'For scene {scene.name}, red dot world coords are {scene.red_dot_wc}\n {e}')
            is_error = True
        iter_time = time.time() - start_time
        return mappings, reverse_mappings, iter_time, is_error  # errx, erry

