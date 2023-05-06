"""Code to apply a rotation to the image"""

import math
from typing import List
import numpy as np
from image_editor.preparations.image import Image

class Rotate(Image):
    """A Class to apply a rotation to the image.
    To provide consistency in the image shape, the removed pixels are replaced by black pixels.
    Inherits from Image class as we create a new Image object"""

    def __init__(self, path: str = '', n_degrees: tuple = (-45,45)) -> None:
        """Constructor of the class. Inherits the attributes from Image
        Args:
            path: the path of the original image
            n_degrees: specifies the range (in degrees) from which we should randomly
                       draw a rotation
            """
        super().__init__(path)
        self.n_degrees = n_degrees

    def generate(self, n: int =2) -> List[Image]:
        '''Generates n images with different rotation degrees'''

        self.load()
        rotations = np.random.uniform(self.n_degrees[0],self.n_degrees[1],n)

        imgs = []

        for rot in rotations:
            rads = math.radians(rot)

            tensor = self.tensor
            rot_img = np.zeros(tensor.shape)

            height = rot_img.shape[0]
            width  = rot_img.shape[1]
            midw,midh = (width//2, height//2)

            if rads >= 0:

                for i in range(rot_img.shape[0]):
                    for j in range(rot_img.shape[1]):
                        x = (i-midh)*math.cos(rads)+(j-midw)*math.sin(rads)
                        y = -(i-midh)*math.sin(rads)+(j-midw)*math.cos(rads)

                        x=round(x)+midh
                        y=round(y)+midw

                        if (x>=0 and y>=0 and x<tensor.shape[0] and  y<tensor.shape[1]):
                            rot_img[i,j,:] = tensor[x,y,:]

            else:
                for i in range(rot_img.shape[0]):
                    for j in range(rot_img.shape[1]):
                        x = (i-midh)*math.cos(-rads)-(j-midw)*math.sin(-rads)
                        y = +(i-midh)*math.sin(-rads)+(j-midw)*math.cos(-rads)

                        x=round(x)+midh
                        y=round(y)+midw

                        if (x>=0 and y>=0 and x<tensor.shape[0] and  y<tensor.shape[1]):
                            rot_img[i,j,:] = tensor[x,y,:]

            image = Image(tensor=rot_img)
            imgs.append(image)

        return imgs
