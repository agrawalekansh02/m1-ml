# Apple M1 ML Test

Some simple test to get `Tensorflow` running on Apple's M1 architecture.

## Getting started

install conda:

```sh
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

install dependencies

```sh
conda install -c apple tensorflow-deps
```

follow commands when upgrading base TensorFlow version

```sh
# uninstall existing tensorflow-macos and tensorflow-metal
python -m pip uninstall tensorflow-macos
python -m pip uninstall tensorflow-metal
# Upgrade tensorflow-deps
conda install -c apple tensorflow-deps --force-reinstall
# or point to specific conda environment
conda install -c apple tensorflow-deps --force-reinstall -n my_env
```

install TensorFlow(s)

```sh
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```