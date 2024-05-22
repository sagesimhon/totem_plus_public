import math
from multiprocessing import get_context

import psutil as psutil

from config import mi_variant
from utils.generation.xml_based.mappings import Coords, RedDot
import os
import json
import time
from datetime import datetime

import matplotlib.pyplot as plt

# import logging
# logging.basicConfig(format='%(asctime)s %(clientip)-15s %(user)-8s %(message)s', level=logging.DEBUG)

# wr = weakref.ref(x)
# gc.get_referents(wr)
#


# compute the number of CPUs we can use based on the machine number of CPS and the memory available
def get_cpus(cpu_spare=1, mem_spare_gb=0.5, memory_usage_required_per_process_gb=.4):
    n_cpus = os.cpu_count() - cpu_spare
    free_memory_gb = psutil.virtual_memory().available/2**30
    max_cpus_from_memory = int((free_memory_gb - mem_spare_gb)/memory_usage_required_per_process_gb)
    if max_cpus_from_memory < 0:
        raise MemoryError(f'Not enough memory: {free_memory_gb}GB available, '
                          f'min required: {memory_usage_required_per_process_gb} ')
    if n_cpus > max_cpus_from_memory:
        print(f'adjusting cpus from {n_cpus} to {max_cpus_from_memory} due to memory constraints')
        n_cpus = max_cpus_from_memory
    factor = 3 if 'llvm' in mi_variant and n_cpus > 12 else 2 if 'llvm' in mi_variant else 1    # split parallelization resources
    return math.ceil((n_cpus or 1) / factor) # If you would like to hardcode the num CPUs used, just change this to return # CPUs desired (int)


def save_mappings(base_name, path, mappings, reverse_mappings, iter_times):
    f = os.path.join(path, f'{base_name}.json')
    print(f"Saving mappings to {f}")

    with open(f, "w") as file:
        json.dump(mappings, file)
    f = os.path.join(path, f'{base_name}_cam_to_tot.json')
    with open(f, "w") as file:
        json.dump(reverse_mappings, file)
    f = os.path.join(path, 'iter_times.json')
    with open(f, "w") as file:
        json.dump(iter_times, file)


def run(fstart='mappings', path_to_xmls=None, requested_file=None, path_to_exp_folder=None, cli_args=None):
    # Caveat: Args functionality relies on run_generation.py being run before this function (since it calls parse args)

    iter_times = []
    path_to_xmls=path_to_xmls or 'XMLs'
    path_to_exp_folder= path_to_exp_folder or '.'
    count_failed = 0
    mappings = {}
    reverse_mappings = {}
    is_parallel = cli_args.p if cli_args else False  # distribute scenes in parallel, each scene is also parallelized
    if requested_file is not None:
        num = requested_file[-11:-4]
        bboxgetter = Coords(requested_file, None)
        reddot_coords = bboxgetter.wc_reddot
        try:
            dot = RedDot(requested_file, bboxgetter.bounding_box, bboxgetter.res_x, bboxgetter.res_y, spp=cli_args.spp)
            print('file ' + num + ' ' + reddot_coords + ":" + str(dot.uv_cam) + str(
                dot.uv_tot))  # x,y (x+ ->, y+ v)
            mappings[str(dot.uv_tot)] = str(dot.uv_cam)
            reverse_mappings[str(dot.uv_cam)] = str(dot.uv_tot)

        except AssertionError as e:
            print('file ' + num + ' ' + reddot_coords + f": Failed: {e}")

    else:
        print("Fetching xml files, this might take some time...", flush=True)
        files_to_test = [fname for fname in os.listdir(path_to_xmls) if
                 fname.startswith(fstart) and not fname.endswith('LO.xml') and os.path.isfile(
                     os.path.join(path_to_xmls, fname))]
        sorted_files_to_test = sorted(files_to_test, key=lambda x: int(x[-11:-4])) #int(x[-8:-4] is the 4 digit 'number' identifier on the file
        try:
            if is_parallel:
                print(f"Generating job params to distribute {len(sorted_files_to_test)} files", flush=True)
                params = [
                    [i, fname, path_to_xmls, path_to_exp_folder, fstart, cli_args.spp] for i, fname in enumerate(sorted_files_to_test)
                ]
                n_cpus = get_cpus() if not cli_args or cli_args.n_cpus <= 0 else cli_args.n_cpus
                print(f'Distributing {len(params)} jobs across {n_cpus} parallel processes', flush=True)

                # even with multiprocessing, there still hang some references that cause memory leaks at each iteration
                # so let's do this in batches to try to clear any memory leaks
                # we terminate (not close) and recreate a new pool every, say, 30 cycles, to avoid overhead
                # from termination and recreation
                batch_size = n_cpus * 100
                param_batches = [params[i: i + batch_size] for i in range(0, len(params), batch_size)]
                start = time.time()
                for i, params in enumerate(param_batches):
                    with get_context("spawn").Pool(n_cpus) as pool:
                        results = pool.starmap(process_file, params)
                        for result in results:
                            _mappings, _reverse_mappings, _iter_time, _is_error = result
                            mappings |= _mappings
                            reverse_mappings |= _reverse_mappings
                            iter_times.append(_iter_time)
                            if _is_error:
                                count_failed += 1
                        save_mappings(fstart, path_to_exp_folder, mappings, reverse_mappings, iter_times)
                        print(f'Terminating process pool {i}', flush=True)
                        pool.terminate()          # try this to avoid memory leaks
                        current_time = time.time()
                        batch_time = (current_time - start) / (i+1)
                        file_time = batch_time/batch_size
                        total_hr = batch_time * len(param_batches) / 60 / 60
                        print(f'{datetime.fromtimestamp(current_time)} -- Progress {i/len(param_batches)*100:.1f}% (batch #{i}/{len(param_batches)}:'
                              f' batch size {len(params)})--- {batch_time} sec/batch, {file_time} sec/file -- total {total_hr}hr', flush=True)
            else:
                start = time.time()
                for i, fname in enumerate(sorted_files_to_test):
                    _mappings, _reverse_mappings, _iter_time, _is_error = process_file(i, fname, path_to_xmls, path_to_exp_folder, fstart, cli_args.spp)
                    mappings |= _mappings
                    reverse_mappings |= _reverse_mappings
                    iter_times.append(_iter_time)
                    if _is_error:
                        count_failed += 1
                    if i % 100 == 0:
                        batch_time = (time.time() - start) /(i+1)
                        total_hr=batch_time*len(sorted_files_to_test) / 60 / 60
                        print(f'Progress ({i/len(sorted_files_to_test)*100:.1f}%) {i} / {len(sorted_files_to_test)} --- {batch_time} sec/file -- total {total_hr}hr', flush=True)
                        save_mappings(fstart, path_to_exp_folder, mappings, reverse_mappings, iter_times)
            save_mappings(fstart, path_to_exp_folder, mappings, reverse_mappings, iter_times)
            f1 = os.path.join(path_to_exp_folder, f'{fstart}.json')
            f2 = os.path.join(path_to_exp_folder, f'{fstart}_cam_to_tot.json')
            with open(f1, "w") as file:
                json.dump(mappings, file)
            with open(f2, "w") as file:
                json.dump(reverse_mappings, file)

            f3 = os.path.join(path_to_exp_folder, 'iter_times.json')
            with open(f3, "w") as file:
                json.dump(iter_times, file)
            '''
            for i, fname in enumerate(sorted_files_to_test):
                # if fname.startswith(fstart) and not fname.endswith('LO.xml') and os.path.isfile(f'{xml_dir}/'+fname):
                #     # fname = f'{xml_dir}/'+fname
                iter_start_time = time.time()

                num = fname[-9:-4]
                print("Trying iter " + str(i) + ", file " + num)
                bboxgetter = Coords(fname, path_to_xmls)
                reddot_coords = bboxgetter.wc_reddot
                try:
                    dot = RedDot(fname, path_to_xmls, bboxgetter.bounding_box, bboxgetter.res_x, bboxgetter.res_y)
                    # print('iter: ' + str(i) + '. file ' + num + ' ' + reddot_coords + ":" + str(dot.uv_cam) + str(dot.uv_tot)) #x,y (x+ ->, y+ v)
                    print('For file' + num + ' ,reddot world coords are ' + reddot_coords + "\n uv_cam, uv_tot: \n" + str(dot.uv_cam) + str(dot.uv_tot)) #x,y (x+ ->, y+ v)
                    mappings[str(dot.uv_tot)] = str(dot.uv_cam)
                    reverse_mappings[str(dot.uv_cam)] = str(dot.uv_tot)
                    # this_data = {str(dot.uv_tot): str(dot.uv_cam)}
                    f1 = os.path.join(path_to_exp_folder, f'{fstart}.json')
                    f2 = os.path.join(path_to_exp_folder, f'{fstart}_cam_to_tot.json')
                    with open(f1, "w") as file:
                        json.dump(mappings, file)
                    with open(f2, "w") as file:
                        json.dump(reverse_mappings, file)
                    del dot
                except AssertionError as e:
                    # print('iter: ' + str(i) + '. file ' + num + ' ' + reddot_coords + f": Failed: {e}")
                    print('For file' + num + ' ,reddot world coords are ' + reddot_coords + "\n uv_cam, uv_tot: \n" + f'Failed: {e}') #x,y (x+ ->, y+ v)
                    count_failed += 1

                iter_end_time = time.time()
                iter_time = iter_end_time - iter_start_time
                iter_times.append(iter_time)
            '''
            print("Failed count: ", count_failed)
            print(str(len(reverse_mappings))+"total mappings found.")
            print(str(len(reverse_mappings)-len(mappings)) + "duplicate uv_tots for distinct uv_cams.")
            plt.bar(range(len(sorted_files_to_test)), iter_times)
            plt.title('Runtime for each iteration (each red dot file in sweep)')
            plt.xlabel('File index')
            plt.ylabel('Time (s)')
            plt.savefig(os.path.join(path_to_exp_folder,'timing_iters.png'))
            plt.close()
        except Exception as e:  # KeyboardInterrupt:
            print(e)
            plt.bar(range(len(iter_times)), iter_times)
            plt.title('Runtime for each iteration (each red dot file in sweep)')
            plt.xlabel('File index')
            plt.ylabel('Time (s)')
            plt.savefig(os.path.join(path_to_exp_folder, 'timing_iters.png'))
            plt.close()
    # Save the dictionary to a file in JSON format
    # with open(f'{fstart}.json', "w") as file:
    #     json.dump(mappings, file)
    return mappings, reverse_mappings


def process_file(i, fname, path_to_xmls, path_to_exp_folder, fstart, spp):
    # if fname.startswith(fstart) and not fname.endswith('LO.xml') and os.path.isfile(f'{xml_dir}/'+fname):
    #     # fname = f'{xml_dir}/'+fname
    iter_start_time = time.time()
    is_error = False
    mappings = {}
    reverse_mappings = {}
    num = fname[-11:-4]
    print("Trying iter " + str(i) + ", file " + num, flush=True)
    bboxgetter = Coords(fname, path_to_xmls)
    reddot_coords = bboxgetter.wc_reddot
    try:
        # import pdb; pdb.set_trace()
        dot = RedDot(fname, path_to_exp_folder, path_to_xmls, bboxgetter.bounding_box, bboxgetter.res_x, bboxgetter.res_y, spp=spp)
        # print('iter: ' + str(i) + '. file ' + num + ' ' + reddot_coords + ":" + str(dot.uv_cam) + str(dot.uv_tot)) #x,y (x+ ->, y+ v)
        print('For file' + num + ' ,reddot world coords are ' + reddot_coords + "\n uv_cam, uv_tot: \n" + str(
            dot.uv_cam) + str(dot.uv_tot), flush=True)  # x,y (x+ ->, y+ v)
        mappings[str(dot.uv_tot)] = str(dot.uv_cam)
        reverse_mappings[str(dot.uv_cam)] = str(dot.uv_tot)
        # this_data = {str(dot.uv_tot): str(dot.uv_cam)}
        # f1 = os.path.join(path_to_exp_folder, f'{fstart}.json')
        # f2 = os.path.join(path_to_exp_folder, f'{fstart}_cam_to_tot.json')
        # with open(f1, "w") as file:
        #    json.dump(mappings, file)
        # with open(f2, "w") as file:
        #    json.dump(reverse_mappings, file)
        del dot
    except AssertionError as e:
        # print('iter: ' + str(i) + '. file ' + num + ' ' + reddot_coords + f": Failed: {e}")
        print('For file' + num + ' ,reddot world coords are ' + reddot_coords + "\n uv_cam, uv_tot: \n" + f'Failed: {e}', flush=True)  # x,y (x+ ->, y+ v)
        is_error = True
    iter_end_time = time.time()
    iter_time = iter_end_time - iter_start_time

    return mappings, reverse_mappings, iter_time, is_error
