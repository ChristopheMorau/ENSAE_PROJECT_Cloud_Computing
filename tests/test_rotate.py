"""Tests for the Rotate functionnality"""

import numpy as np

from image_editor.transformations.rotation import Rotate
from image_editor.preparations.image import Image


TEST_IMAGE = "run_local/images/téléchargement.jpg"

def test_rotation():
    """Check if Rotation perform as expected
    If rotation works well, all 4 corners extremities should be black pixels
    And the rotated image tensor should be of the same shape than the original image tensor"""
    image_original = Image(path = TEST_IMAGE)
    image_original.load()
    image_rotate = Rotate(path = TEST_IMAGE)
    imgs = image_rotate.generate()
    img = imgs[0]
    top_right_pix = img.tensor[0, -1]
    top_left_pix = img.tensor[0, 0]
    bot_right_pix = img.tensor[-1, -1]
    bot_left_pix = img.tensor[-1, 0]
    assert np.array_equal(top_right_pix,np.zeros(3))
    assert np.array_equal(top_left_pix,np.zeros(3))
    assert np.array_equal(bot_right_pix,np.zeros(3))
    assert np.array_equal(bot_left_pix,np.zeros(3))
    assert img.tensor.shape == image_original.tensor.shape
