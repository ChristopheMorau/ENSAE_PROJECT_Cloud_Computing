"""Test for parent class Image"""

from image_editor.preparations.image import Image


def test_loading():
    """test for parent class Image"""
    image = Image(path = "run_local/images/téléchargement.jpg")
    image.load()
    assert image.tensor.shape == (183, 275, 3)
    