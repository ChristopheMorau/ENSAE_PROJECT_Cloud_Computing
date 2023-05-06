"""Code to modify the brightness of an image """

from typing import List
import numpy as np
from image_editor.preparations.image import Image

class Brightness(Image):
    """Class containing a method to generate, from one image,
    multiple copies with different brightness."""

    def __init__(self, path: str) -> None:
        """Constructor of the class. Inherits the attributes from Image
        Args:
            path: the path of the original image
            m: number of images to be generated"""
        super().__init__(path)

    def generate(self, n : int =2) -> List[Image]:
        '''Generates n images with different brightness'''
        self.load()
        brs = np.random.uniform(0.1,1,n)
        imgs = []
        for brightness in brs:
            tensor = brightness*self.tensor
            image = Image(tensor=tensor)
            imgs.append(image)

        return imgs
