import cv2
import numpy as np
from PIL import Image
import math
import time


def dft(input_img, rows, cols):
    output_img = np.zeros((rows, cols), complex)
    for u in range(0, rows):  # Moving along rows
        for v in range(0, cols):  # Moving along cols
            for x in range(0, rows):  # Evaluation loop
                for y in range(0, cols):  # Evaluation loop
                    output_img[u][v] += input_img[x][y] * np.exp(-1j * 2 * math.pi * (u * x / rows + v * y / cols))
            output_img[u][v] = output_img[u][v] / (rows*cols)
    return output_img


def pixel_normalization(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            unorm_image[i, j] = ((unorm_image[i, j] - pxmin)/(pxmax-pxmin))*255

    norm_image = unorm_image
    return norm_image


def center_image(image):
    centered_image = np.zeros((rows, cols))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            centered_image[i, j] = image[i, j]*((-1)**((i-1)+(j-1)))

    return centered_image


if __name__ == "__main__":
    path = r'Dataset/64x64.jpg'
    img = cv2.imread(path, 0)
    rows = img.shape[0]
    cols = img.shape[1]

    ori_image = img
    ori_image = center_image(ori_image)
    start_time = time.time()
    image = dft(ori_image, rows, cols)
    # image = np.fft.fft2(ori_image) # comparing to standard library

    image = abs(image)
    norm_image = pixel_normalization(image)

    print("___ %s seconds ___" % (time.time() - start_time))

    im = Image.fromarray(norm_image)

    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("64x64dft.jpg")
