import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pywt
import random
def fuseCoeff(cooef1, cooef2):

    cooef = np.maximum(cooef1, cooef2)
    return cooef

def Haar_DWT(channel: int, img1, img2):

    wavelet = pywt.Wavelet('haar')


    cooef1 = [pywt.wavedec2(img1[:, :, c], wavelet, level=1) for c in range(channel)]
    cooef2 = [pywt.wavedec2(img2[:, :, c], wavelet, level=1) for c in range(channel)]

    fused_cooef = [[],[],[]]
    for index in range(channel):
        for i in range(len(cooef1[index])):
            if (i == 0):
                fused_cooef[index].append(fuseCoeff(cooef1[index][0], cooef2[index][0]))
            else:
                c1 = fuseCoeff(cooef1[index][i][0], cooef2[index][i][0])
                c2 = fuseCoeff(cooef1[index][i][1], cooef2[index][i][1])
                c3 = fuseCoeff(cooef1[index][i][2], cooef2[index][i][2])
                fused_cooef[index].append((c1, c2, c3))

    # print(len(fused_cooef))
    fused_image = [0,0,0]
    for index in range(channel):
        fused_image[index] = pywt.waverec2(fused_cooef[index], wavelet)
        fused_image[index] = 255 * fused_image[index] / np.max(fused_image[index])
        fused_image[index] = fused_image[index].astype(np.uint8)

    new_fused_image = np.zeros((534, 400, 3), dtype=np.uint8)
    for index in range(channel):
        for i in range(534) :
            for j in range(400) :
                new_fused_image[i][j][index] = fused_image[index][i][j]
    return new_fused_image

if __name__ == '__main__':

    add_image = cv.imread('final_rainbow_extracted.jpg', flags=cv.COLOR_RGB2HLS)
    target_image = cv.imread('xidian.jpg', flags=cv.COLOR_RGB2HLS)
    add_image = cv.resize(add_image, (target_image.shape[1], target_image.shape[0]))

    channel = target_image.shape[2]
    final_image = Haar_DWT(channel, target_image, add_image)
    cv.imwrite('success_rainbow_extracted.jpg', final_image)
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()
