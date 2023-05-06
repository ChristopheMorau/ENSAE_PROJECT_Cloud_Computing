"""Tests for the flipping functionnality"""

from image_editor.transformations.flip import Flip

TEST_IMAGE = "run_local/images/téléchargement.jpg"


def test_flip_vertically():
    """Unit test to check if the image behaves as expected when flipped vertically"""
    image_flip = Flip(path = TEST_IMAGE, both=False)
    image_flip.load()
    last_element_c1 = image_flip.tensor[-1,0].copy()
    imgs = image_flip.generate()
    first_element_c1_new = imgs[0].tensor[0,0]
    assert (first_element_c1_new == last_element_c1).all()

def test_flip_horizontally():
    """Unit test to check if the image behaves as expected when flipped horizontally"""
    image_flip = Flip(path = TEST_IMAGE, vectically=False, both=False)
    image_flip.load()
    last_element_l1 = image_flip.tensor[0,-1].copy()
    imgs = image_flip.generate()
    first_element_l1_new = imgs[0].tensor[0,0]
    assert (first_element_l1_new == last_element_l1).all()

def test_flip_both():
    """Unit test to check if setting both=True successfully creates a list with a vertical and 
    horizontal flip"""
    image_flip = Flip(path = TEST_IMAGE)
    image_flip.load()
    last_element_c1 = image_flip.tensor[-1,0].copy()
    last_element_l1 = image_flip.tensor[0,-1].copy()
    imgs = image_flip.generate()
    assert len(imgs) == 2
    first_element_l1_new = imgs[1].tensor[0,0]
    assert ( (first_element_l1_new == last_element_l1).all()
    | (first_element_l1_new == last_element_c1).all() )
    first_element_l1_new = imgs[0].tensor[0,0]
    assert ( (first_element_l1_new == last_element_l1).all()
    | (first_element_l1_new == last_element_c1).all() )