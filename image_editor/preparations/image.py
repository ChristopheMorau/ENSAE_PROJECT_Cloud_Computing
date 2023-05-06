"""A class to define our image object"""
import numpy as np
import matplotlib.pyplot as plt   # type: ignore


class Image:
    """Image object, created from frontend request (or path in run_local setting)"""

    def __init__(self, path: str = '', tensor: np.ndarray = np.ones(1)) -> None:
        """Constructor of the class
        Args:
            path : defines where to find the image
        """
        self.path: str = path
        self.tensor: np.ndarray = tensor

    def load(self):
        """Method to load the image from the specified path"""
        if len(self.path) > 0:
            self.tensor = plt.imread(self.path)



    def print_image(self, cmap: str = '') -> None:
        """Displays the image with matplotlib"""
        if len(cmap) == 0:
            plt.imshow(self.tensor)
        else:
            plt.imshow(self.tensor, cmap=cmap)
        plt.show()

    def save(self, path) -> None:
        """Method to save the image and update the path"""
        self.path = path
        if self.tensor.max() > 1:
            self.tensor = self.tensor/255
        plt.imsave(path, self.tensor)
