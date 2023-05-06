"""Tests for the image basic colors generating functionnality"""

from image_editor.transformations.colors import Colors

TEST_IMAGE = "run_local/images/téléchargement.jpg"


def test_red_image():
    """test to check if we generate the red layer of the image within the Image_colors class """
    img=Colors(path=TEST_IMAGE)
    img.load()
    image=img.red_image
    assert (image[:,:,1]==0).all()
    assert (image[:,:,2]==0).all()

def test_green_image():
    """test to check if we generate the green layer of the image within the Image_colors class"""
    img=Colors(path=TEST_IMAGE)
    img.load()
    image=img.green_image
    assert (image[:,:,0]==0).all()
    assert(image[:,:,2]==0).all()

def test_blue_image():
    """test to check if we generate the blue layer of the image within the Image_colors class"""
    img=Colors(path=TEST_IMAGE)
    img.load()
    image=img.blue_image
    assert (image[:,:,0]==0).all()
    assert(image[:,:,1]==0).all()

def test_grayscale_image():
    "test to check if the image is grayscale "
    img=Colors(path=TEST_IMAGE)
    img.load()
    image=img.gray_image
    h,w=len(image), len(image[0])
    for i in range(h):
        for j in range(w):
            assert image[i][j] in range(0,266)
    assert len(image.shape)==2
