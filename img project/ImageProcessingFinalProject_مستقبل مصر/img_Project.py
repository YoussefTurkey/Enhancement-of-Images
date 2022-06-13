# our Libreries
from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters

# We will upload new hand sign image with PIL 'Python Imaging Librery'.
# We will show it as original image.
stack = []
for i in range(5):
    print('Image is loading...')
    img = Image.open(f'img_{i+1}.jpg')
    stack.append(img)
    # print(stack)
for j in range(5):
    stack[j].show()

    # We will convert the image to grayscale image, and save it as 'ok_gray.jpg'.
    # We will show it as gray image.
stack_Gray = []
for i in range(5):
    print('Graying an image is loading...')
    imgGray = stack[i].convert('L')
    stack_Gray.append(imgGray)
for j in range(5):
    stack_Gray[j].show()
for k in range(5):
    new_imgGray = stack_Gray[j].save(f'img_{i+1}_gray.jpg')

    # We will detect the image border, and save it as 'img_edge.jpg'.
    # We will show it as bordered image.
stack_Edge = []
for i in range(5):
    print('Detecting an edge of image is loading...')
    find_edges = stack[i].filter(ImageFilter.FIND_EDGES)
    stack_Edge.append(find_edges)
for j in range(5):
    stack_Edge[j].show()
for k in range(5):
    new_imgEdge = stack_Edge[j].save(f'img_{i+1}_edge.jpg')

    # We will use the sharpen method firstly, and save it as 'sharpen_img.jpg'.
    # We will show it as sharpen image.
stack_Sharp = []
for i in range(5):
    print('Sharping an image is loading...')
    sharpen_img = stack[i].filter(ImageFilter.SHARPEN)
    stack_Sharp.append(sharpen_img)
for j in range(5):
    stack_Sharp[j].show()
for k in range(5):
    new_imgSharpe = stack_Sharp[j].save(f'img_{i+1}_sharpen.jpg')

    # We will use after that smoothing method, and save it as 'smooth_img.jpg'.
    # We will show it as smoothed image.
stack_Smooth = []
for i in range(5):
    print('Smoothing an image is loading...')
    smooth_img = stack[i].filter(ImageFilter.SMOOTH)
    stack_Smooth.append(smooth_img)
for j in range(5):
    stack_Smooth[j].show()
for k in range(5):
    new_imgSmooth = stack_Smooth[j].save(f'img_{i+1}_smooth.jpg')

    # We will convert grayscale image to binary image
stack_Binary = []
for i in range(5):
    print('Convert grayscale image to binary image is loading...')
    img_gray = stack[i].convert('L')
    binary_img = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
    stack_Binary.append(binary_img)

    # We will convert the image from 'RGB' to 'RGBA'
    # We will make the image transparency by segment the background
    # , and save it as 'img_without_bg.png'.
    # We will show it as image without backgroud.
stack_BG = []
for j in range(5):
    print('Removing a background of an image is loading...')
    img_BG = stack[i].convert('RGBA')
    for item in img_BG.getdata():
        if item[:3] == (255, 255, 255):
            stack_BG.append((255, 255, 255, 0))
        else:
            stack_BG.append(item)
    # There is an error that "too many data entries" here
    img_BG.putdata(stack_BG)
for k in range(5):
    stack_BG[k].show()
for m in range(5):
    new_img_without_BG = stack_BG[m].save(f'img_{i+1}_without_BG.jpg')

    # Creating class called 'Enhancement'


class Enhancement:
    # Guessian Function
    def gaussianMask(m, n, sigma):
        gaussianMask = np.zeros((m, n))
        m = m//2
        n = n//2
        for x in range(-m, m+1):
            for y in range(-n, n+1):
                x1 = sigma*(2*np.pi)**2
                x2 = np.exp(-(x*2+y*2)/(2*sigma*2))
                gaussianMask[x+m, y+n] = (1/x1)*x2
        return gaussianMask

        # Convolution Function
    def convolute(img, filter):
        filter = filters.convolve(img, mode='constant', cval=0)
        img_h = img.shape[0]
        img_w = img.shape[1]
        filter_h = filter.shape[0]
        filter_w = filter.shape[1]
        h = filter_h//2
        w = filter_w//2
        img_conv = np.zeros(img.shape)
        for i in range(h, img_h-h):
            for j in range(w, img_w-w):
                sum = 0
                for m in range(filter_h):
                    for n in range(filter_w):
                        sum = sum+filter[m][n]*img[i-h+m][j-w+n]
                    img_conv[i][j] = sum
        return img_conv

        # Sobil Function
    def sobil(img):
        laplacian = cv.Laplacian(img, cv.CV_64F)
        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)  # x
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)  # y

        plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

        return sobelx, sobely

        # Perwitt Function
    def prewitt(image):
        Kx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        Ky = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
        px = ndimage.filters.convolve(image, Kx)
        py = ndimage.filters.convolve(image, Ky)
        G = np.hypot(px, py)
        m = G.max()
        G *= image.max() / m
        return G

        # Non Max Suppression Function
    def non_max_suppression(img, angle):
        M, N = img.shape
        Non_max = np.zeros((M, N))

        for i in range(1, M-1):
            for j in range(1, N-1):
                # 0 degree
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    b = img[i, j+1]
                    c = img[i, j-1]
                    # 45 degree
                elif (22.5 <= angle[i, j] < 67.5) or (-157.5 <= angle[i, j] < -112.5):
                    b = img[i+1, j+1]
                    c = img[i-1, j-1]
                    # 90 degree
                elif (67.5 <= angle[i, j] < 112.5) or (-112.5 <= angle[i, j] < -67.5):
                    b = img[i+1, j]
                    c = img[i-1, j]
                    # 135 degree
                elif (112.5 <= angle[i, j] < 157.5) or (-67.5 <= angle[i, j] < -22.5):
                    b = img[i+1, j-1]
                    c = img[i-1, j+1]

                    # non_max_suppression
                if (img[i, j] >= b) and (img[i, j] >= c):
                    Non_max[i, j] = img[i, j]
                else:
                    Non_max[i, j] = 0
        return Non_max

        # Double Threshold Function
    def DoubleThreshold(image, lowratio, highratio):
        img = img.copy()
        highthreshold = img.max() * highratio
        lowthreshold = highthreshold * lowratio
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                pixel = image[i][j]
                if(pixel >= highthreshold):
                    pixel = 255
                elif (pixel <= lowthreshold):
                    pixel = 0
                else:
                    pixel = 125
                img[i][j] = pixel
        return img

        # Edge Linking Function
    def EdgeLinking(image):
        img = image.copy()
        rows, cols = img.shape
        for row in range(1, rows-1):
            for col in range(1, cols-1):
                pixel = img[row][col]
                if(pixel > 127 and pixel < 130):
                    if (image[row][col+1] >= 130):
                        pixel = 255
                    elif (image[row+1][col+1] >= 130):
                        pixel = 255
                    elif (image[row-1][col+1] >= 130):
                        pixel = 255

                    elif (image[row+1][col] >= 130):
                        pixel = 255
                    elif (image[row-1][col] >= 130):
                        pixel = 255

                    elif (image[row+1][col-1] >= 130):
                        pixel = 255
                    elif (image[row-1][col-1] >= 130):
                        pixel = 255
                    elif (image[row][col-1] >= 130):
                        pixel = 255
                    else:
                        pixel = 0

                    img[row][col] = pixel
        return img

        # Canny Function
    def canny(image):
        img = image.copy()
        lowerthreshold = 0.05
        higherthreshold = 0.1
        img = stack[i](img)
        img = Enhancement.convolute(img, Enhancement.gaussianMask(5, 1.4))
        img, angles = Enhancement.sobel(img)
        img = Enhancement.non_max_suppression(img, angles)
        img = Enhancement.DoubleThreshold(img, lowerthreshold, higherthreshold)
        img = Enhancement.EdgeLinking(img)
        img.show([image, img])
        return img


    # Calling Functions
g = Enhancement.gaussianMask(3, 3, 1)
c = Enhancement.convolute(stack[i], g)
plt.imshow(stack[i], cmap='gray')
plt.figure()
plt.imshow(stack[i], cmap='gray')
plt.show
s = Enhancement.sobil(stack[i])
d = Enhancement.DoubleThreshold(stack[i])
e = Enhancement.EdgeLinking(stack[i])
cn = Enhancement.canny(stack[i])
p = Enhancement.prewitt(stack[i])
nms = Enhancement.non_max_suppression(stack[i], 90)
