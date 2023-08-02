import os.path

import cv2
import numpy as np


def output(img, file_name, path='E:\\File\\tmp\\origin'):
    file_name = path + file_name
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    cv2.imwrite(file_name, img)


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def main():
    img = cv2.imread("E:\\File\\tmp\\blur.bmp")
    gauss_img = cv2.GaussianBlur(img, (51, 51), sigmaX=5)
    output(gauss_img, "0926.bmp")


if __name__ == '__main__':
    main()
