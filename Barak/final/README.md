# Hydra Regeneration Cell Split Detection and Analysis

This project aims to detect and analyze cell split events in hydra regeneration using Python. The hydra is a small aquatic organism known for its remarkable regenerative abilities. By studying the process of cell splitting during hydra regeneration, we can gain insights into cellular mechanisms and potentially apply them to other fields such as tissue engineering and regenerative medicine.

<p align="center">
  <img src="2021_07_26_pos_4_V_726.png" width="70%" alt="logo">
</p>

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Files](#files)
- [Output](#output)
- [Results](#results)

## Introduction

Hydra regeneration is a complex biological process that involves the splitting of cells to form new tissues and organs. This project focuses on developing a Python-based solution to detect and analyze cell split events in hydra regeneration. By leveraging image processing techniques and machine learning algorithms, we aim to accurately identify and track cell split events in hydra samples.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

A simple run:

```bash
python main.py -dl directories.txt -p --create_clips --polar --df_hist
```

An example of a line in `directories.txt`:

```
\\phhydra\phhydraB\Analysis\users\Liora\Movie_Analysis\2021_07_26\2021_07_26_pos4
```

Overall:

```bash
usage: main.py [-h] [-d DIR] [-r FRAME_RANGE] [-o OUTPUT] [-dl DIRECTORIES_LIST] [--create_csv]
               [--create_clips] [-p] [--polar] [--df_hist]

options:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Directory containing the images
  -r FRAME_RANGE, --frame_range FRAME_RANGE
                        The range of frames to include in the video. Default is all frames.
  -o OUTPUT, --output OUTPUT
                        The output folder
  -dl DIRECTORIES_LIST, --directories_list DIRECTORIES_LIST
                        A file with a list of directories to process
  --create_csv          Create new cells.csv with split_id
  --create_clips        Create video clips of each split and full video
  -p, --pickle          Store / Load from pickle
  --polar               Generate polar plots of the cells around the split
  --df_hist             Display histogram of the angle between the split dipole and the fibers
```

## Data

The hydra regeneration dataset used in this project consists of a collection of high-resolution video frames captured during the regeneration process, Segmentation of each frame and data tables.  
The code depends on this data to be in their default locations.

## Files

| File Name                 | Description                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| `main.py`               | The main Python script for detecting and analyzing cell split events in hydra regeneration. |
| `directories.txt`       | A text file containing a list of directories to process for cell split detection.           |
| `directories_times.csv` | A table containing the time interval for each video.                                        |
| `requirements.txt`      | A file specifying the required dependencies for running the project.                        |
| `data_statistics.py`    | A Python library for statistical analysis of cell split patterns.                           |
| `data_manager.py`       | A Python library for managing data files/(images, segmemtation, csv)                        |
| `cell_split.py`         | A Python library for detecting cell split events.                                           |
| `split.py`              | A Python library with the split class.                                                      |
| `utils.py`              | A Python library with general utilities.                                                    |

## Output

By default, the output will be saved in the runned directory. It can be change by `-o` flag.   
The output depends on the flags that were used. In total:

```bash
└── output/
    ├── graphs
    │   ├── area
    |   |   ├── fold
    |   |   |   ├── i_t_n.png
    |   |   |   ├── ...
    |   |   └── regular
    │   ├── aspect_ratio
    │   ├── average_brightness
    │   ├── circle_identify
    │   ├── ellipse_identify
    │   ├── neighbors
    │   ├── perimeter
    │   ├── Qrr_projection
    │   ├── Qrt_projection
    │   ├── main_axis_angle.png
    │   └── fibers_angle.png
    ├── <video_identifier>
    │   ├── graphs
    │   ├── splits
    |   │   ├── <split_video>.avi
    |   |   ├── ...
    │   ├── video_frames
    |   │   ├── <split_image>.png
    |   │   ├── ...
    │   ├── <full_video>.avi
    ├── ...
```

Where:

* `<split_video>` is the video of the split.
* `<split_image>` is the image of the split.
* `<full_video>` is the full video.
* `...` means that there are more files/folders as the above.
* `graphs` under `<video_identifier>` is the same structre as `graphs` under `output`.
* Each folder under `graphs` have the same structure.
* `i_t_n.png` is the polar graph at time `t` and with `n` data points.

## Results

The results of this project include:

- Visualization of detected cell split events
- Statistical analysis of cell split patterns
- Insights into the hydra regeneration process
