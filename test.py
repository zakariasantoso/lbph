import numpy as np
from skimage import data
import matplotlib.pyplot as plt
# %matplotlib inline
   
image = data.camera()  
type(image)
# numpy.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray')