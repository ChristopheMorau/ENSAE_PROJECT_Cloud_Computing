"""Test module for Brightness"""
from image_editor.transformations.brightness import Brightness


def test_brightness():
    """test brightness"""
    image = Brightness(path = "run_local/images/téléchargement.jpg")
    imgs = image.generate()
    for i , image in enumerate(imgs):
        image.save(f"run_local/images/br_{i}.jpg")
