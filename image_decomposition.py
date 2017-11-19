from Signal_image import *
from numpy import transpose, array, uint8, zeros, append, insert

g = array([0.5, 0.5])
h = array([0.5, -0.5])


def decomposition(image):
    # get HF part of image
    hf = convolution(image, h)
    # get LF part of image
    lf = convolution(transpose(image), g)
    return hf, lf


def convolution(image, vector):
    """Convolution of image and vector
    VERY SLOW"""
    newim = []
    h = vector
    for line in image:
        # insert zero to the start and to the end
        line = insert(line, 0, 0)
        line = append(line, 0)
        newline = []
        for i in range(len(line)-2):
            m = line[i:i+2]*h
            s = sum(m)
            newline.append(s)
        newim.append(newline)
    return newim


def composition(hf_image, lf_image):
    """Composition of images"""
    result = zeros((len(hf_image), len(hf_image[0])))
    for i in range(len(hf_image)):
        for j in range(len(hf_image[0])):
            result[i][j] = hf_image[i][j] + lf_image[i][j]
    return result


im = load_image('Lenna.jpg')
hf, lf = decomposition(im)
Image.fromarray(uint8(hf)).show(title='LOW frequency')
Image.fromarray(uint8(lf)).show(title='HIGH frequency')
image = composition(hf, lf)
Image.fromarray(image).show()
