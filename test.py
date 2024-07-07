import numpy as np
import matplotlib
matplotlib.use("QtAgg")
# matplotlib.use("agg")
# matplotlib.use("nbAgg")
import matplotlib.pyplot as plt

# matplotlib.use("WebAgg")

"""
Load image as sample
"""
# Load an image and normalize it
my_object_RGB = plt.imread('imgs/PiotrZakrzewski_5197202.png')  # RGB image
fig, ax = plt.subplots()
ax.imshow(my_object_RGB)
fig.show()
# plt.show()