"""Code to change the color of images"""

from typing import List
import numpy as np
from image_editor.preparations.image import Image


def rgb2gray(image:np.ndarray=np.ones(1))->np.ndarray:
    """ convert image from RGB to grayscale using the weighted method
    grayscaling converting methods doesn't seem to work perfectly for png image format """
    image_height = len(image)
    image_width = len(image[0])
    gray_image = np.empty([image_height, image_width], dtype=np.uint8)

    for i in range(image_height):
        for j in range(image_width):
            gray_image[i][j]=(image[i][j][0])*0.2989+ (image[i][j][1])*0.587 +(image[i][j][2])*0.114
    return gray_image

def image_2_blue(image:np.ndarray=np.ones(1))->np.ndarray:
    """Extract the blue layer of the image"""
    blue_image=image.copy()
    blue_image[:,:,0]=0
    blue_image[:,:,1]=0
    return  blue_image

def image_2_green(image:np.ndarray=np.ones(1))->np.ndarray:
    """Extract the green layer of the image"""
    green_image=image.copy()
    green_image[:,:,0]=0
    green_image[:,:,2]=0
    return  green_image


def image_2_red(image:np.ndarray=np.ones(1))->np.ndarray:
    """Extract the red layer of the image"""
    red_image=image.copy()
    red_image[:,:,1]=0
    red_image[:,:,2]=0
    return  red_image

class Colors(Image):
    "class to generate basic image colors"

    def __init__(self, path: str = '')->None:
        """ constructor of the class. Inherit the attributes from Image """
        super().__init__(path)
        self.load()
        if self.path[-3:len(path)]=='png':
            self.save(self.path[0:len(self.path)-4]+'.jpg')
            self.load()
        self.gray_image = rgb2gray(self.tensor)
        self.red_image=image_2_red(self.tensor)
        self.blue_image=image_2_blue(self.tensor)
        self.green_image=image_2_green(self.tensor)


    def generate(self)-> List[Image]:
        """ generate a list of 4 basic colors images"""
        gray_image=Image(tensor=self.gray_image)
        green_image=Image(tensor=self.green_image)
        blue_image=Image(tensor=self.blue_image)
        red_image=Image(tensor=self.red_image)
        imgs=[gray_image,green_image,blue_image,red_image]
        return imgs
