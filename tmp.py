import numpy as np
import skimage
import matplotlib.pyplot as plt


im = skimage.io.imread("/Users/lironst/Downloads/view7_max.tif")
idx = 0
plt.figure(figsize=(6, 6), layout="tight")
plt.imshow(im[idx], cmap="gray")
plt.title(f"Slice {idx}")
plt.show()
pass
