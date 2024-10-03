import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")
# matplotlib.use("agg")
# matplotlib.use("nbAgg")


# matplotlib.use("WebAgg")


"""
Load image as sample
"""
# Load an image and normalize it
my_object_RGB = plt.imread('imgs/PiotrZakrzewski_5197202.png')  # RGB image
fig, ax = plt.subplots()
ax.imshow(my_object_RGB)
# plt.show()
fig.canvas.draw()
# plt.ion()
fig.canvas.flush_events()

plt.show()
# fig.canvas.draw()
