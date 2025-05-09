import os
import numpy as np
import matplotlib.pyplot as plt
import napari

# viewer.layers["Labels"].data.shape

viewer = napari.Viewer()
viewer.open(
		path=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2\data",
		layer_type="image",
		name="train_2",
)
viewer.add_image()
# todo: continue from here
# todo: perform labeling to several images, run pixel classification in napari and make sure it's good.
# todo: If so, perform statistics. If not, run from ilastik.
