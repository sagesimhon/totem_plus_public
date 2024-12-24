# totem_plus
![alt text](https://github.com/sagesimhon/totem_plus_public/blob/main/sample_other_shapes.png)
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
run_generation.py (STEP 1)      # TOP LEVEL wrapper that sets up a new experiment to create a new correspondences dataset and supporting visualization files, following rules specified by the sweep ranges and other py args. 
                                # Output data saved in {path_to_exp}/.

run_nn.py (STEP 2)              # TOP LEVEL wrapper that handles everything related to training, testing, logging, inference, &
                                # visualizing a training experiment. Everything is saved in the specified experiment folder, and 
                                # the associated correspondences dataset of that exp is used as training data. Outputs saved in path_to_exp/nn/{training_exp_name}/.

run_unwarping_image_based.py (STEP 3)       # TOP LEVEL wrapper to reconstruct the original image from an unverified input image, using the specified exp. Outputs saved in path_to_exp/{unwarping_exp_name}.

run_detection.py (STEP 4)       # TOP LEVEL wrapper to run the detection stage; runs detection and evaluation on all or specified reconstructions, and all outputs (heatmaps, psnr scores) are saved in subfoldersa within path_to_exp/unwarping_exp_name/.

Other supporting files:

meshes/                         # mesh .obj files used in synthetic scene generation, including cornell box parts and totem shapes

models/                         # MLP model(s) for predicting uv_cam <-> uv_tot refraction mappings 
train.py                        # Training file for the MLP, supporting training, testing, visualization (inference is also run from here)

utils/                          # Helpers for all 4 phases as well as processing data for your experiments: correspondences dataset generation, NN training, reconstruction, and detection, and manipulation
    generation/                 # Correspondences generation related helpers
    nn/                         # NN related helpers for data pre and post processing
    reconstruction/             # Helpers for unwarping totem image 
    detection/                  # Helpers for running inconsistency detection + metrics
    manip/                      # Support for automated colorpatch manipulation on desired data 

config.py                       # For setting the rendering backend variant and input/output data paths
environment.yml                 # Dependencies 
run_command.sh                  # Helper for running remote experiments on shell
run_*_machine_distributed.sh    # Support to run large experiments remotely and in parallel, on GPUs or CPUs
run_kill_jobs.sh                # Automatically stop all remote processes that are running via run_*_machine_distributed.sh  
wavelets.py                     # Visualization of correspondence dataset frequency heatmaps and wavelet transforms of input images

data/                           # This is where input data (XML files) for mapping generation and unwarping is held 
    image_to_unwarp/            # Reference images used for unwarping
    xml_starters/               # xml files of scenes used as starting points for red dot sweep, unwarping, 
                                # color correlation, or other rendering/visualization

                                
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
   - Change the variable in line 6 of the code `input_data_base_path = {absolute path to your project dir - i.e. 'totems_plus_public' if you retain the repo name in your clone}/data`
   <br>


4. (Variants) Optional: If you would like to manually install the compiler used by Mitsuba's rendering backend, LLVM, instead of via `environment.yml`:
   - On Linux: `sudo dnf install llvm`. On mac: `brew install llvm`. 
   - `export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.so` (e.g. `export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM.so`)
   - If you would like to use a different rendering variant, see `mitsuba` docs for a list of rendering variants, and change accordingly in `config.py`.

## How to run

This code supports the full, 4 stage pipeline of our work outlined above.

#### To generate a correspondences dataset:
   1. Run `python run_generation.py --exp_folder {your_experiment_name} --res {image resolution}`
      - Note that larger resolutions will take a very long time to run locally (res 728 took 2-3 days on my M1 Mac). For this reason, the following options are available:
        - You can run a pruned sweep and subsample test points with the optional argument --sweep_density (fraction indicating % of test points sampled, default 1.0. I recommend using a small number like 0.1 to first run a quick sample experiment to test your environment and understand the output structure). We found that reduced sweep density (as low as 0.07) yields quantitatively comparable reconstruction and detection results, in a fraction of the time. 
        - Another means for a fast test, or to target only a desired region, you can additionally edit the sweep ranges using --x/y_min and --x/y_max 
        - Support for remote runs on a distributed system of GPUs can be found in run_generation_gpu_machine_distributed.sh
        - There is an optional boolean --p argument indicating whether to use Python parallelization. Due to job splitting overhead, it only yielded slight improvements in speed, so we would recommend using one or a combination of the above alternative workarounds.
      - For all custom settings, including totem shape, see `utils/generation/get_mappings_args.py`

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
1. Run `python run_nn.py --exp_folder {experiment folder containing the mappings dataset, generated above} --res_x {resolution of the training set, i.e. resolution used in run_generation.py} --res_y {resolution of the training set, i.e. resolution used in run_generation.py} --run_extension {name for this NN experiment}`. 
See `utils/nn/get_nn_args.py` for more training, model architecture, and dataset-related settings.
<br>

Results will be saved to <br>
`{absolute_path_to_your_data_dir}/{your_experiment_name}/nn/{value for run_extension arg}.` <br>
Results include: <br>
`models/model_*.pt`, the saved trained model on each epoch <br>
`metadata.json`, some metadata about the model to be used later for reconstruction (unwarping) <br>
`pred_vs_truth_full.png`, a scatter plot of all datapoints/preds, stats on final train/test/val errors. Note this scatterplot is generally not very readable due to the sheer number of datapoints (WIP). <br>
`random_sample_visual_*.png`, ...which is why we include visualizations on tiny subsets of mid-training predictions at various epochs. <br>
`training_curve.png`, training curve plot <br>

Note that we also support Tensorboard logging. You can visualize results on Tensorboard by running 
`tensorboard --logdir={exp_folder}`.

#### To unwarp (reconstruction)

0. Save desired (totem-less) unverified images in a folder under `path_to_repo/data/images_to_unwarp` 
1. Run `python run_unwarping_image_based.py --exp_folder {same as above} --nn_folder {same as run_ext above} --res_x {must be consistent with the resolution of the mappings dataset for the associated exp} --res_y {must be consistent with the resolution of the mappings dataset for the associated exp} --im_folder images_to_unwarp` 
<br>

Results will be saved to 
`{absolute_path_to_your_data_dir}/{your_experiment_name}/{image_name_*_unwarped.png}.` <br>

[//]: # (## Custom scenes for test sweeping )

[//]: # (If you would like to add your own custom XML file for sweeping &#40;for example, the totem might be located)

[//]: # (in a different position or the camera extrinsics might be different&#41;, I will add easier instructions for this. )

[//]: # (For now just directly change the file `unit_test_starter_res*.xml` where * indicates the desired res in consideration.)

# Sample Results
![alt text](https://github.com/sagesimhon/totem_plus_public/blob/main/sample_results.png)
