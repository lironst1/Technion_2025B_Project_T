July, 2025. Liron Stettiner. lironst1@gmail.com.

## Table of Contents

- [Introduction](#Introduction)
- [Cellpose](#Cellpose)
- [Code](#Code)

## Introduction

The project is focused on identifying Hydra nuclei by measuring β-catenin signals using the Airyscan microscope. This
classification is then used in order to characterize changes in intensity, average size, distance and other
cellular-level statistics.
![[2025_03_05__View1__4_beta_cat25X_TL_T9_C1.png]]

## Cellpose

Cellpose ([Nature article](https://doi.org/10.1038/s41592-020-01018-x), [GitHub and installation (make sure to install the GPU version of PyTorch)](https://github.com/MouseLand/cellpose), [documentation](https://cellpose.readthedocs.io/en/latest/), [GUI documentation](https://cellpose.readthedocs.io/en/latest/gui.html), [YouTube tutorial](https://www.youtube.com/watch?v=5qANHWoubZU))
is a neural network used for the segmentation of biological cells (BTW, the model also supports 3D images, so perhaps we
could extend this research to use the raw data without max projection).

Cellpose has both a Python module and a GUI. I used the GUI to adjust model evaluation parameters by loading several
test images and fine-tuning the parameters (such that false-positives will be relatively small). Later, I used these
parameters in the Python module together with the rest of the code for automation.

To run the Cellpose GUI, follow the instructions in [GitHub](https://github.com/MouseLand/cellpose). Install it in your
Python environment:

```bash
pip install cellpose
```

Then run:

```bash
python -m cellpose
```

![[Pasted image 20250705122447.png]]
To load an image, click `Ctrl+L` or go to `File -> Load Image`.
You can change the parameters under the `Segmentaion -> additional settings` section and the
`Image filtering -> cusom filter settings` section. Importantly, make sure to set the average cell diameter (in pixels).
In the past, there used to be multiple models you could choose between, but today there is only one available.
To run the model, click the `Segmentaion -> run CPSAM` button.

## Code

| File Name     | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `__init__.py` | The module initialization script.                                  |
| `__cfg__.py`  | A configuration file for all user-defined settings and parameters. |
| `main.py`     | The main Python script.                                            |
| `utils.py`    | A Python library with general utilities.                           |

TBD...
