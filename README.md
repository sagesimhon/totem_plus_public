# totem_plus
![alt text](https://github.com/sagesimhon/totem_plus/blob/main/sample_other_shapes.png)
Main contribution of my Master's thesis. 

(Please note the following instructions may be slightly deprecated, as I am in the process of cleaning+documenting the most up-to-date iteration of this project for public release).


This repo contains the following structure: 
```angular2html
data/                           # This is where input data (XML files) for mapping generation and unwarping is held 
    iamge_to_unwarp/            # Reference images used for unwarping
    xml_starters/               # xml files of scenes used as starting points for red dot sweep, unwarping, 
                                # color correlation, or other rendering/visualization

meshes/                         # mesh .obj files used to create cornell boxes in all xml files in this repo
models/                         # contains the uv_cam <-> uv_tot prediction MLP model 
train.py                        # All supporting training code for the NN 

utils/                          # All necessary helpers for arg parsing, mapping generation, color correlation, NN 
                                # training, testing, inference, visualization, and unwarping
    generation/                 # Mapping generation related helpers
    nn/                         # NN related helpers 
    reconstruction/             # Unwarping related helpers 

The 3 files you will be running are: 
run_mapping_generation.py       # TOP LEVEL wrapper that sets up a new experiment for a new mappings datset.
                                # Integrates helpers + mappings.py into a desired mapping generation specified by the  
                                # sweep ranges and other python args. Generates color correlation. NN and unwarping 
                                # results will be saved in this experiment folder as well.

run_nn.py                       # TOP LEVEL wrapper that handles everything related to training, testing, logging, &
                                # visualizing a NN experiment. Everything saved in the experiment folder of the 
                                # corresponding mappings dataset.

run_unwarping.py                # Unwarp an image using the dataset and NN model specified 
                                
```

# Workflow
## Dependencies
1. Create a new conda environment with the necessary dependencies installed:

   - `conda env create -f environment.yml`

   - Activate the env with `conda activate totems` 

   *NOTE* For a faster conda resolver, you can install the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) before running these commands


2. Make a local data directory for your experiments (preferably outside the project directory to avoid IDE indexing slowdowns. These directories will contain hundreds of thousands of XML files for the mappings sweep): 

   - e.g.``mkdir /Users/{your user}/data``
   or 
   `mkdir data` in the desired location 


3. Update the config file with the dependency from (2), and with your project name: 

   - Change the variable in line 5 of the code `output_data_base_path = {absolute_path_to_your_data_dir}`
   <br> 
   - Change the variable in line 4 of the code `xml_data_base_path = {absolute path to your project dir - this will likely be called totems_plus}/data` <br>


4. (Variants) Optional: If you would like to manually install the compiler used by Mitsuba's rendering backend, LLVM, instead of from `environment.yml`:
   - on Linux: `sudo dnf install llvm`. On mac: `brew install llvm`. 
   - `export DRJIT_LIBLLVM_PATH=/path/to/libLLVM.so` (e.g. `export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM.so`)
   - If you would like to use a different rendering variant, see `mitsuba` docs for a list of rendering variants, and change accordingly in `config.py`.

## How to run

This code supports generation (i.e. running a sweep and creating + saving a mappings dataset), training, and reconstruction (unwarping). 

#### To generate a mappings dataset: 
   1. (Optional): Change the sweep range in L12-14 of `run_generation.py`. Default settings are fine if you would 
like to sweep in the region that has been demonstrated so far in meetings. Note that the scene ranges from [-1,1] in all 3 dimensions.
   2. Run with `python run_generation.py --exp_folder {your_experiment_name} --res {choose 728 or 256}`. Note that larger resolutions will take a very long time to run locally (res 728 took 2-3 days on my M1 Mac). 256 is reasonable.
   <br> Other custom settings, including "n" the granularity of the sweep, are described in `utils.generation.get_mappings_args.py`
   <br> For a fast run to just see working results, change `--n 50`
   3. If you would like to disable parallelization or change the number of CPUs used or memory available, edit the relevant lines in 
   `utils.generation.mappings_toplevel_helpers.py`. This involves the bool `is_parallel` and the arguments passed in on the call to `get_cpus`. Make sure to change the spare memory availability param in `get_cpus` (`mem_spare_gb`), if you want to make the most of your machine's resources. 
<br>

That's it! Results will be saved to `{absolute_path_to_your_data_dir}/{your_experiment_name}/`.  <br>
Results include <br>
`mappings.json`, a dictionary of mappings `{uv_tot : uv_cam}`,  <br>
`mappings_cam_to_tot.json`, a reverse dictionary `{uv_cam : uv_tot}` 
overcoming the many-to-one data loss in the former dictionary <br>
`space_covered.png`, a visual of the space of valid mappings found (any pixel that is purple represents a valid mapping found),  <br>
`color_corr.png`, a 1:1 (data lost) color correlation, <br>
`timing_iters.png`, a barplot of runtime/datapoint for each test point in the sweep <br>
plus a few other minor things.. (todo)

#### To train a NN 
1. Run `python run_nn.py --exp_folder {experiment folder containing the mappings dataset, generated above} --run_ext {name for this NN experiment}`. 
See `utils.nn.get_nn_args.py` for more training settings.
<br>

Results will be saved to <br>
`{absolute_path_to_your_data_dir}/{your_experiment_name}/nn/{value for run_ext arg}.` <br>
Results include: <br>
`model.pt`, the saved trained model <br>
`metadata.json`, some metadata about the model used later for reconstruction (unwarping) <br>
`pred_vs_truth_full.png`, a plot visualizing train + test results. <br>

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
