"""Tests for the shift functionnality"""

from image_editor.transformations.shift import Shift


TEST_IMAGE = "run_local/images/téléchargement.jpg"

def test_shift_vertically():
    """Check if vertical shifts perform as expected"""
    image_shift = Shift(path = TEST_IMAGE, pick_randomly=False)
    image_shift.load()
    first_pixel = image_shift.tensor[0, 0]
    imgs = image_shift.generate()
    img = imgs[0]
    assert (first_pixel == img.tensor[60, 0]).all()

def test_shift_horizontally():
    """Check if horizontal shifts perform as expected"""
    image_shift = Shift(path = TEST_IMAGE, vertically=False, pick_randomly=False)
    image_shift.load()
    first_pixel = image_shift.tensor[0, 0]
    img = image_shift.generate()[0]
    assert (first_pixel == img.tensor[0, 60]).all()

def test_shift_randomly():
    """Check if randomly selected shifts performs as expected"""
    image_shift = Shift(path = TEST_IMAGE, pick_randomly=True)
    image_shift.load()
    first_pixel = image_shift.tensor[0, 0]
    imgs = image_shift.generate()
    n_pixel: int = image_shift.n_pixels
    assert len(imgs) == 2
    img = imgs[-1]
    assert ( (first_pixel == img.tensor[0, n_pixel]).all() |
             (first_pixel == img.tensor[n_pixel, 0]).all())
