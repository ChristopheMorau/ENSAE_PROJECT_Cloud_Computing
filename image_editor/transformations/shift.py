"""Code to apply a translation to the image"""

from typing import List
import numpy as np
from image_editor.preparations.image import Image

class Shift(Image):
    """A Class to apply a translation to the image. The image is moved by x pixels in one direction.
        To provide consistency in the image shape, the first x pixels in that direction are replaced
        by black pixels, the last x pixels are removed
        Inherits from Image class as we create a new Image object"""

    def __init__(self, path: str = '', vertically: bool = True, n_pixels: int = 60,
                pick_randomly: bool = True) -> None:
        """Constructor of the class. Inherits the attributes from Image
        Args:
            path: the path of the original image
            vertically: specifies if the shift should be vertical (True) or horizontal (False)
                        Ignored if pick_randomly = True
            n_pixels: speciefies by how many pixels should be performed the shift
                       Ignored if pick_randomly = True
            pick_randomly: if True, vertically and n_pixels are set randomly
            """
        super().__init__(path)
        self.vertically = vertically
        self.n_pixels = n_pixels
        self.pick_randomly = pick_randomly

    def _choose_randomly(self) -> None:
        """Picks randomly a direction and a number of pixels for the shift
           n_pixels is selected such that at most half of the image is translated"""

        choice: int = np.random.binomial(1, 1/2)
        if choice == 1:
            self.vertically = True
            tensor_shape: int = self.tensor.shape[0]

        else:
            self.vertically = False
            tensor_shape = self.tensor.shape[1]
        self.n_pixels = np.random.randint(0, int(tensor_shape/2))

    def _shift_vertically(self) -> Image:
        """Applies a vertical shift to the image"""

        tensor_vertical_shape: int = self.tensor.shape[0]
        tensor_horizontal_shape: int = self.tensor.shape[1]
        if self.n_pixels >= tensor_vertical_shape:
            raise ValueError("n_pixels should be of value smaller than the image vertical size")

        if self.tensor.shape[2]==4:
            black_array = np.zeros( (self.n_pixels, tensor_horizontal_shape, 4), 'int' )
        if self.tensor.shape[2]==3:
            black_array = np.zeros( (self.n_pixels, tensor_horizontal_shape, 3), 'int' )

        tensor: np.ndarray = np.concatenate((black_array, self.tensor), axis=0)
        tensor = tensor[:tensor_vertical_shape, :, :]
        img: Image = Image(tensor = tensor)
        return img

    def _shift_horizontally(self) -> Image:
        """Applies a horizontal shift to the image"""

        tensor_vertical_shape: int = self.tensor.shape[0]
        tensor_horizontal_shape: int = self.tensor.shape[1]
        if self.n_pixels >= tensor_horizontal_shape:
            raise ValueError("n_pixels should be of value smaller than the image horizontal size")

        if self.tensor.shape[2]==4:
            black_array: np.ndarray  = np.zeros( (tensor_vertical_shape, self.n_pixels, 4), 'int')
        if self.tensor.shape[2]==3:
            black_array = np.zeros( (tensor_vertical_shape, self.n_pixels, 3), 'int')

        tensor: np.ndarray = np.concatenate((black_array, self.tensor), axis=1)
        tensor = tensor[:, :tensor_horizontal_shape, :]
        img: Image = Image(tensor = tensor)
        return img

    def generate(self, n: int=2) -> List[Image]:
        """Applies the required shift to the image"""
        self.load()
        imgs = []
        if self.pick_randomly:
            for iteration in range(n):
                self._choose_randomly()
                if self.vertically:
                    img = self._shift_vertically()
                else:
                    img = self._shift_horizontally()
                imgs.append(img)
        else:
            if self.vertically:
                imgs.append(self._shift_vertically())
            else:
                imgs.append(self._shift_horizontally())
        return imgs
