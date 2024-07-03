import numpy as np
import imageio.v2 as imageio
import os

folder = 'imgs_for_gif'
filename = 'myplot'
indices = np.arange(0,100,2, dtype=int)
filenames = [filename + f'_{i}.png' for i in indices]

images = []
for filename in filenames:
    path = os.path.join(folder, filename)
    images.append(imageio.imread(path))
imageio.mimsave('movie.gif', images, fps=2, loop=0)
