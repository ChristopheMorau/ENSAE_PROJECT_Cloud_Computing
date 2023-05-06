"""Code to add noise to the image"""

from typing import List
import numpy as np
from image_editor.preparations.image import Image

class GaussianNoise(Image):
    """Class containing a method to add gaussian noise to an image."""

    def __init__(self, path: str) -> None:
        """Constructor of the class. Inherits the attributes from Image
        Args:
            path: the path of the original image
            m: number of images to be generated"""
        super().__init__(path)

    def generate(self, mean=0, sigma=0.1) -> List[Image]:
        '''Generates a noisy image'''
        self.load()
        image = self.tensor

        row,col,ch= image.shape
        sigma = sigma*image.max()

        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)

        noisy_tensor = image + gauss
        noisy_tensor = noisy_tensor.clip(0,255)
        noisy = Image(tensor = noisy_tensor)
        return [noisy]
