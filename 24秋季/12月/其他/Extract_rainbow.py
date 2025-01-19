import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def show_image(img) :

    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img, cmap='gray')
    plt.title('GS Image'), plt.xticks([]), plt.yticks([])

if __name__ == '__main__':

    image = cv.imread('rainbow.jpg',0)
    g_image  = cv.GaussianBlur(image,(5,5),1.5)
    # g_image = image
    #开始边缘检测
    show_image(g_image)
    new_image = cv.Canny(g_image,50 ,100 )

    plt.subplot(133), plt.imshow(new_image, cmap='gray')
    plt.title('Final Image'), plt.xticks([]), plt.yticks([])
    plt.show()
