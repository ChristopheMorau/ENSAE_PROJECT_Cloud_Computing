"""A script to try things locally, does the same as main.py but uses images from run_local/images
while main.py gets the images sent by the frontend (app.py)"""


from typing import List
from image_editor.preparations.image import Image
from image_editor.transformations.rotation import Rotate
from image_editor.transformations.shift import Shift
from image_editor.transformations.flip import Flip

image_rotated: Rotate = Rotate(path = "run_local/images/téléchargement.jpg")
imgs: List[Image] = image_rotated.generate()
print(imgs[0].tensor.shape)
imgs[0].save("run_local/images/test_rotation.jpg")

image_shift: Shift = Shift(path = "run_local/images/téléchargement.jpg")
imgs = image_shift.generate()
imgs[0].print_image()

#Flips an image
image_flipped: Flip = Flip(path = "run_local/images/téléchargement.jpg")
imgs = image_flipped.generate()
imgs[1].print_image()
