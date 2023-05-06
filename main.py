"""Entry point for the backend code"""

import os
from typing import List
from image_editor.preparations.image import Image
from image_editor.transformations.flip import Flip
from image_editor.transformations.colors import Colors
from image_editor.transformations.brightness import Brightness
from image_editor.transformations.gaussian_noise import GaussianNoise
from image_editor.transformations.rotation import Rotate
from image_editor.transformations.shift import Shift


# transfo_required: List[str] = ['Shift','Flip', 'Colors','Brightness','GaussianNoise','Rotate']
INPUTPATH: str = 'static/images_input/'
OUTPUTPATH: str = 'static/images_output/'

def generate_all(transfo_required: List[str]) -> None:
    """Function in charge of handling all the transformations required by the user
    Args: transformations_required : list of transformations required, corresponds to class names
    """

    input_images: List[str] = [INPUTPATH + image for image in os.listdir(INPUTPATH)]
    current_output_images: List[str] = [OUTPUTPATH + image for image in os.listdir(OUTPUTPATH)]
    if len(current_output_images) > 0:
        for output_image in current_output_images:
            os.remove(output_image)
    for path in input_images:
        for transfo in transfo_required:
            module = globals()[transfo]
            transformater = module(path)
            imgs: List[Image] =  transformater.generate()
            for i, img in enumerate(imgs):
                img.save(OUTPUTPATH+path[20:]+'_'+transfo+'_'+str(i)+'.jpg')
