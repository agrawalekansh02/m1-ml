python -m pip uninstall tensorflow-macos
python -m pip uninstall tensorflow-metal

conda install -c apple tensorflow-deps --force-reinstall

python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
