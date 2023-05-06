"""Code to flip horizontally of vertically the image"""

from typing import List
import numpy as np

from image_editor.preparations.image import Image

class Flip(Image):
    """Class to flip horizontally of vertically the image"""


    def __init__(self, path: str = '', vectically: bool = True, both: bool = True) -> None:
        """Constructor of the class. Inherits the attributes from Image
        Args:
            path: the path of the original image
            vertically: Determines if we split the image vertically (True) or horizontally (False)
                        Ignored if random_flip = True
            both: when true, we return both the vertical and horizontal flip """
        super().__init__(path)
        self.vertically = vectically
        self.both = both

    def _flip_horizontally(self) -> Image:
        """Applies an horizontal flip to the image"""
        tensor: np.ndarray = np.fliplr(self.tensor)
        img = Image(tensor = tensor)
        return img

    def _flip_vertically(self) -> Image:
        """Applies a vertical flip to the image"""
        tensor: np.ndarray = np.flipud(self.tensor)
        img = Image(tensor = tensor)
        return img

    def generate(self) -> List[Image]:
        """Applies the required flip to the image"""
        self.load()
        if self.both:
            imgs = [self._flip_vertically(), self._flip_horizontally()]
        else:
            imgs = []
            if self.vertically:
                img = self._flip_vertically()
                imgs += [img]
            else:
                img = self._flip_horizontally()
                imgs += [img]
        return imgs
