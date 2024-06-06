# totem_plus
![alt text](https://github.com/sagesimhon/totem_plus/blob/main/sample_other_shapes.png)
Main contribution of my Master's thesis. 

(Please note the following instructions may be slightly deprecated, as I am in the process of cleaning+documenting the most up-to-date iteration of this project for public release).

Our approach for image verification comprises 4 stages which we often refer to throughout this project: 
1. Correspondences dataset generation
2. NN training
3. Reconstruction
4. Detection

This repo contains the following structure: 
```angular2html
The four files you will use for the bare steps are: 
run_generation.py (STEP 1)      # TOP LEVEL wrapper that sets up a new experiment to create a new correspondences dataset.
                                # Integrates helpers + mappings.py into a desired mapping generation specified by the  
                                # sweep ranges and other python args. Generates color correlation. NN and unwarping 
                                # results will be saved in this experiment folder as well.

run_nn.py (STEP 2)              # TOP LEVEL wrapper that handles everything related to training, testing, logging, &
                                # visualizing a training experiment. Everything is saved in the experiment folder of the 
                                # corresponding correspondences dataset, which was generated in step 1.

run_unwarping.py (STEP 3)       # Reconstruct the original image from an unverified input image, using the specified exp

run_detection.py (STEP 4)       # Top level detection stage

Other supporting files: 

data/                           # This is where input data (XML files) for mapping generation and unwarping is held 
    image_to_unwarp/            # Reference images used for unwarping
    xml_starters/               # xml files of scenes used as starting points for red dot sweep, unwarping, 
                                # color correlation, or other rendering/visualization

meshes/                         # mesh .obj files used in synthetic scene generation, include cornell box parts and totem shapes

models/                         # contains the MLP model(s) for predicting uv_cam <-> uv_tot refraction mappings 
train.py                        # Top level training file for the MLP, supporting training, testing, inference, and visualization

utils/                          # Helpers for all phases of the method including: correspondences generation, color correlation, data preprocessing/NN training, reconstruction, and detection 
    generation/                 # Correspondences generation related helpers
    nn/                         # NN related helpers for data pre and post processing
    reconstruction/             # Helpers for unwarping totem image 
    detection/                  # Helpers for running inconsistency detection + metrics
    manip/                      # Support for automated colorpatch manipulation on desired data 

config.py                       # For setting the rendering backend variant and input/output data paths
environment.yml                 # Dependencies 
run_command.sh                  # Helper for running remote experiments on shell
run_*_machine_distributed.sh    # Support to run large experiments remotely and in parallel, on GPUs or CPUs
run_kill_jobs.sh                # Failsafe stop all remote processes running via run_*_machine_distributed.sh  
run_nn.py                       # Top level NN training file
wavelets.py                     # Visualization of correspondence frequency heatmaps and wavelet transforms of input images

                                
```

# Workflow
## Dependencies
1. Create a new conda environment with the necessary dependencies installed:

   - `conda env create -f environment.yml`

   - Activate the env with `conda activate totems` 

   *NOTE* For a faster conda resolver, you can install the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) before running these commands

    - Set the `DRJIT_LIBLLVM_PATH` environment variable to specify the path to `libLLVM.dylib`, e.g. `export DRJIT_LIBLLVM_PATH=/Users/sage/miniconda3/envs/totems/lib/libLLVM.18.1.dylib`    


2. Make a local data directory for your experiments: 

   - e.g.``mkdir /Users/{your user}/experiments``
   or 
   `mkdir experiments` in the desired location 


3. Update the config file with the dependency from (2), and with your project name: 

   - Change the variable in line 7 of the code `output_data_base_path = {absolute_path_to_your_experiments_dir}`
   <br> 
   - Change the variable in line 6 of the code `input_data_base_path = {absolute path to your project dir - i.e. 'totems_plus' if you keep the same name in your clone}/data` <br>


4. (Variants) Optional: If you would like to manually install the compiler used by Mitsuba's rendering backend, LLVM, instead of via `environment.yml`:
   - On Linux: `sudo dnf install llvm`. On mac: `brew install llvm`. 
   - `export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.so` (e.g. `export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM.so`)
   - If you would like to use a different rendering variant, see `mitsuba` docs for a list of rendering variants, and change accordingly in `config.py`.

## How to run

This code supports the 4 stages of our work outlined above.

#### To generate a correspondences dataset:
   1. Run `python run_generation.py --exp_folder {your_experiment_name} --res {minimum res supported is 256}`
      - Note that larger resolutions will take a very long time to run locally (res 728 took 2-3 days on my M1 Mac). For this reason, the following options are available:
        - You can run a pruned sweep and subsample test points with the optional argument --sweep_density  --sweep_density (fraction indicating % of test points sampled, default 1.0 but use a small number like 0.1 if you want to run a quick test first). This yields comparable reconstruction and detection results in a fraction of the time. 
        - For a fast test or to target only a desired region, you can additionally edit the sweep ranges using --x/y_min and --x/y_max 
        - Support for remote runs on a distributed system of GPUs can be found in run_generation_gpu_machine_distributed.sh
        - There is an optional boolean --p argument indicating whether to use Python parallelization. Due to job splitting overhead, it only yielded slight improvements in speed, so we would recommend using one of the above alternative workarounds.
      - For all custom settings, including totem shape, see `utils.generation.get_mappings_args.py`

[//]: # (   3. If you would like to disable parallelization or change the number of CPUs used or memory available, edit the relevant lines in )

[//]: # (   `utils.generation.mappings_toplevel_helpers.py`. This involves the bool `is_parallel` and the arguments passed in on the call to `get_cpus`. Make sure to change the spare memory availability param in `get_cpus` &#40;`mem_spare_gb`&#41;, if you want to make the most of your machine's resources. )

That's it! Results will be saved to `{absolute_path_to_your_data_dir}/{your_experiment_name}`.  <br>
Results include <br>
`mappings.json`, a dictionary of mappings `{uv_tot : uv_cam}`,  <br>
`mappings_cam_to_tot.json`, a reverse dictionary `{uv_cam : uv_tot}` 
avoiding the many-to-one data loss in the former dictionary <br>
`failures.json`, containing test points wherein the sweep failed to find a valid refraction <br>
Depending on system level settings, you may also find some files related to runtime/logging stats including `progress.json`, `iter_times.json`, `timing_iters.png`

To qualitatively verify your results, run `python run_generation.py --exp_folder {name_used_above} --res {number_used_above} --action map_corr`.
This will save an image of the 1:1 color correlation (and its corresponding np array). <br>


#### To train a NN to predict correspondneces
1. Run `python run_nn.py --exp_folder {experiment folder containing the mappings dataset, generated above} --res {resolution of the training set, i.e. resolution used in run_generation.py} --run_extension {name for this NN experiment}`. 
See `utils.nn.get_nn_args.py` for more training, model architecture, and dataset-related settings.
<br>

Results will be saved to <br>
`{absolute_path_to_your_data_dir}/{your_experiment_name}/nn/{value for run_extension arg}.` <br>
Results include: <br>
`models/model_*.pt`, the saved trained model on each epoch <br>
`metadata.json`, some metadata about the model to be used later for reconstruction (unwarping) <br>
`pred_vs_truth_full.png`, a scatter plot of all datapoints/preds, stats on final train/test/val errors. Note this scatterplot is generally not very readable due to the sheer number of datapoints. <br>
`random_sample_visual_*.png`, ...which is why we include visualizations on tiny subsets of mid-training predictions at various epochs. <br>
`training_curve.png`, training curve plot <br>

Note that we also support Tensorboard logging, and you can visualize results on Tensorboard by running 
`tensorboard --logdir={exp_folder}`.

#### To unwarp (reconstruction)

1. Run `python run_unwarping.py --exp_folder {same as above} --nn_folder {same as run_ext above} --res {728 or 256, must be consistent with the res of the mappings dataset that the NN trained on} --n_out {50117 for res 728, else anything less than 3000}` 
<br>

Results will be saved to 
`{absolute_path_to_your_data_dir}/{your_experiment_name}/{smiley_face_*_unwarped.png}.` <br>

## Custom scenes for test sweeping 
If you would like to add your own custom XML file for sweeping (for example, the totem might be located
in a different position or the camera extrinsics might be different), I will add easier instructions for this. 
For now just directly change the file `unit_test_starter_res*.xml` where * indicates the desired res in consideration.

# Sample Results
![alt text](https://github.com/sagesimhon/totem_plus/blob/main/sample_results.png)
