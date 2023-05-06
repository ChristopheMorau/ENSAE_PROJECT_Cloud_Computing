"""Test for GaussianNoise"""

from image_editor.transformations.gaussian_noise import GaussianNoise

def test_gaussian_noise():
    """test for gaussian noise"""
    image = GaussianNoise(path = "run_local/images/téléchargement.jpg")
    img = image.generate()
    img[0].save(f"run_local/images/gn_.jpg")
