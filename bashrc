# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/afs/csail.mit.edu/u/s/simhon/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/afs/csail.mit.edu/u/s/simhon/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/afs/csail.mit.edu/u/s/simhon/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/afs/csail.mit.edu/u/s/simhon/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export DRJIT_LIBLLVM_PATH=/data/vision/billf/implicit_scenes/simhon/miniconda-envs/totem/lib/libLLVM-14.so
cd /data/vision/billf/implicit_scenes/simhon/
conda activate /data/vision/billf/implicit_scenes/simhon/miniconda-envs/totem
export HOME=/data/vision/billf/implicit_scenes/simhon
# need to set home dir outside of afs, since afs mount seems to be come corrupt.
# matplot, and drjit save files in home directory and there is no easy way to change that.
# eventually, when the filesystem mount become corrupt, the process fails.
# here is an error example:

# Matplotlib created a temporary cache directory at /tmp/matplotlib-d_jb8hqo because the default path (/afs/csail.mit.edu/u/s/simhon/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
# Matplotlib is building the font cache; this may take a moment.
# jit_kernel_write(): could not write compiled kernel to cache file "/afs/csail.mit.edu/u/s/simhon/.drjit/c1a001ea8c957bc1a7ed2ccd861bb63a.cuda.bin.tmp": Permission denied
# jit_kernel_write(): could not write compiled kernel to cache file "/afs/csail.mit.edu/u/s/simhon/.drjit/c1a001ea8c957bc1a7ed2ccd861bb63a.cuda.bin.tmp": Permission denied
# jit_kernel_write(): could not write compiled kernel to cache file "/afs/csail.mit.edu/u/s/simhon/.drjit/c1a001ea8c957bc1a7ed2ccd861bb63a.cuda.bin.tmp": Permission denied
# jit_kernel_write(): could not write compiled kernel to cache file "/afs/csail.mit.edu/u/s/simhon/.drjit/c1a001ea8c957bc1a7ed2ccd861bb63a.cuda.bin.tmp": Permission denied