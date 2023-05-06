import numpy as np

from image_editor.transformations.rotation import Rotate


TEST_IMAGE = "run_local/images/téléchargement.jpg"

image_rotate = Rotate(path = TEST_IMAGE)
imgs = image_rotate.generate()
img = imgs[0]
top_right_pix = img.tensor[0, -1]
top_left_pix = img.tensor[0, 0]
bot_right_pix = img.tensor[-1, -1]
bot_left_pix = img.tensor[-1, 0]

assert np.array_equal(top_right_pix,np.zeros(3))
assert (top_right_pix == np.zeros(3)).all()
print(top_left_pix)
print(bot_right_pix)
print(bot_left_pix)
