import numpy as np
import cv2


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)
    return matrix


def simplest_cb(img, percent=1):
    """Apply Simplest Color Balance algorithm
       Reimplemented based on https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc"""
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)


# if __name__ == '__main__':
#     img = cv2.imread('C:/Users/ChihYu/Desktop/ToNCKU_imagedata/FS5_20191108/MS_L4/FS5_G010_MS_L4TWD97_20191108_030233'
#                      '/FS5_G010_MS_L4TWD97_20191108_030233.tif')
#     out = simplest_cb(img)
#     cv2.imshow("after", out)
#     cv2.waitKey(0)
