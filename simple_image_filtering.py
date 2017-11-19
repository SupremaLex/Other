from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy import *
from PIL import Image
from matplotlib import pyplot as plt


def load_image(path):
    """Load image"""
    image = Image.open(path).convert('L')
    image.show(title='Original image')
    image = asarray(image, dtype=double)
    return image


def fft(image):
    """Return fft of image and show spectrum"""
    fft_image = fftshift(fft2(image))
    spectrum = fft_image*1.0/len(fft_image)
    spectrum = abs(spectrum)
    Image.fromarray(uint8(20*log(spectrum))).show()
    return fft_image


def ifft(fft_image):
    """Return result of inverse ifft(fft(image))=image"""
    image = ifftshift(fft_image)
    image = ifft2(image)
    image = uint8(real(image))
    Image.fromarray(image).show()
    return image


def ideal_filter(fft_image, filter_type='HF'):
    """Create filter that cut circle from fft(image)"""
    n = len(fft_image)
    m = len(fft_image[0])
    h = zeros((n, m), dtype=complex)
    #max_ = max(abs(fft_image.ravel()))
    for i in range(n):
        for k in range(m):
            h[i][k] = complex(fft_image[i][k].real, fft_image[i][k].imag) * (sqrt((i-n//2)**2+(k-m//2)**2) < 80)
            #h[i][k] = abs(fft_image[i][k]) > max_/700
    # if high freq filter
    if filter_type == 'HF':
        return h
    # else low freq filter
    else:
        return ones((n, m)) - h


def gauss_noise(path):
    """Create an image with gauss noise"""
    image = Image.open(path).convert('L')
    mask = random.poisson(image)
    i = Image.fromarray(uint8(image+mask))
    i.save('Gauss.jpg', 'JPEG')


def salt_and_pepper(path):
    """Create an image with s&p noise"""
    salt_pepper = Image.open(path).convert('L')
    mask = random.poisson()
    for x in range(salt_pepper.size[0]):
        for y in range(salt_pepper.size[1]):
            n = random.random_integers(0, 20)
            if n == 19:
                salt_pepper.putpixel((x, y), (255,))
            elif n == 0:
                salt_pepper.putpixel((x, y), (0,))
    salt_pepper.show()
    salt_pepper.show('Noisy')
    salt_pepper.save('Salt_and_pepper.jpg')


def s_p(image):
    """Other version of s&p noise creating"""
    # This mode makes no effort to avoid repeat sampling. Thus, the
    # exact number of replaced pixels is only approximate.
    out = image.copy()

    # Salt mode
    num_salt = ceil(
        0.05 * image.size * 0.5)
    coords = [random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 255

    # Pepper mode
    num_pepper = ceil(
        0.05 * image.size * 0.5)
    coords = [random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out


if __name__ == '__main__':
    im2 = load_image('SP.jpg')
    fft_im2 = fft(im2)
    h2 = ideal_filter(fft_im2)
    im2 = ifft(fft_im2*h2)
    fft(im2)


