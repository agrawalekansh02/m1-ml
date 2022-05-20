# Apple M1 ML Test

Some simple test to get `TensorFlow` running on Apple's M1 architecture.

Hint: it works really well

![Result](Screen%20Shot%202021-12-21%20at%201.12.31%20AM.png)

## Installation

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

## Tests

- `mnist_dnn.py` -> sample deep neural network performing classification on `mnist` dataset
- `mnist_cnn.py` -> sample convultional neural network performing classfication on `mnist` dataset

## Further Reading

[ways to to make tf architectures](https://pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/)
[seq2seq chatbot](https://medium.com/swlh/how-to-design-seq2seq-chatbot-using-keras-framework-ae86d950e91d)
