
<div align="center">

## Ion counting and temperature determination of Coulomb-crystallized laser-cooled ions in traps using convolutional neural networks

<!-- [![Static Badge](https://img.shields.io/badge/arXiv-ToBeFilled-B31B1B.svg)](https://to-be-filled) -->
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/yanningyin/Coulomb-crystal-CNN/blob/main/LICENSE)

</div>

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Description

This repository contains code for ion counting and temperature determination of Coulomb-crystallized laser-cooled ions in traps using convolutional neural networks (CNNs).

It includes mainly two parts: 1. Generation of simulated images of Coulomb crystals under given conditions. 2. Training and evaluation of (various) CNN models for ion counting and temperature determination of crystals.


## Features
- GPU acceleration possible with [Openmm](https://openmm.org/) framework for image generation and [Pytorch](https://pytorch.org/get-started/locally/) for training and evaluation of CNN models.

-  Fast generation of a large number of images with given ranges of ion number and temperature as a training dataset.

- Various CNN models are available, with or without pre-trained weights, for training.


## Usage

### Install the environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yanningyin/Coulomb-crystal-CNN.git
   ```
2. **Navigate to the root directory**:

   ```bash
   cd your/own/path/to/Coulomb-crystal-CNN
   ```

3. **Create the Environment with conda**:

   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the Environment**:

   ```bash
   conda activate cccnn
   ```

### Image Generation
1. Navigate to directory: `cd image_generation`
2. Define parameters according to your experimental conditions in `PaulTrapSim.cpp`
3. Compile it after modification: `make PaulTrapSim` 
3. Generate multiple images with different numbers of ions and temperatures, e.g.:
   ```bash
   ./runMultiSim.sh 100 200 1 5 20 1
   ```
   where the arguments "100 200 1 5 20 1" correspond to the start, end, increment of number of ions and the start, end, increment of temperature to be simulated, respectively, and should be adapted based on your needs.

4. Images and logs created during the generation of each image are saved in a folder named like "numIons_100_200_1_targetT_5_20_1".

### Image Classification
#### Run locally

1. Navigate to the directory: `cd image_classification`
2. **Train a CNN model**

The python script `cnn-one-label.py` uses command line interface (CLI) to specify arguments. Type in `python3 cnn-one-label.py --help` to see all the parameters and options.

In the following example, the first argument specifies the path to the images for training, the '-m' argument accepts the name of the CNN model, '-w' means the pretrained weights of the model will be loaded, '-rgb' means the images will be converted to rgb mode, '-l t' means the label for classification will be temperature, '-rN' and '-rT' specifies the range of number of ions and temperatures, according to which the images are loaded and the model will be trained to classify.

   ```bash
   python3 cnn-one-label.py your/path/to/images/for/training \
           -m "alexnet" \
           -w -rgb -l t \
           -rN "range(100, 200, 1)" -rT "range(5, 20, 1)"
   ```

   The trained model will be saved in the directory `output` as a file ending with ".pth". A '.txt' file containing all the relevant information during training will be saved with the same name.

3. **Evaluate a CNN model**

Similar to training a model, the CLI specifies the arguments for evaluating a model on images. '-md test' means the model will switch to evaluation mode instead of training mode, '-mp' specifies the path to the model ending with '.pth'.

   ```bash
   python3 cnn-one-label.py your/path/to/images/for/evaluation\
        -md test \
        -mp output/t_5_15_1_alexnet_model_20240731000248.pth \
        -rgb \
        -l t
   ```
4. The evaluation results will be shown like this:
```bash
Image: example.tif, predicted class: 9, time used: 0.014 s
```


#### Submit jobs to a server (Slurm)

- For image generation, adapt the script `image_generation/script-generate-images.sh`
- For image classification, adapt the script `image_classification/scripts/train-model.sh`
- Submit a job using `sbatch` + script
- Logs will be saved in `slurm_logs/` directory

## Project Structure

The directory structure looks like this:

```
.
├── image_classification
│   ├── CLI.py
│   ├── cnn-one-label.py
│   ├── preprocessing.py
│   └── scripts
│       ├── eval-model.sh
│       └── train-model.sh
├── image_generation
│   ├── hist2img.m
│   ├── Makefile
│   ├── PaulTrapSim.cpp
│   ├── runMultiSim.sh
│   └── script-generate-images.sh
├── .gitignore
├── LICENSE
├── README.md
└── environment.yml
```


## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeatureName`
5. Submit a pull request.


<!-- ## Citation

To be filled. -->



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.