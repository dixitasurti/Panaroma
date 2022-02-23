#!/usr/bin/env python
# coding: utf-8

# In[53]:


"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here
    imgArr = np.array(img)
    imgPadded = np.pad(imgArr, 1, mode='constant')
    m,n = imgPadded.shape
    denoise_img = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [imgPadded[i-1, j-1],
                    imgPadded[i-1, j],
                    imgPadded[i-1, j + 1],
                    imgPadded[i, j-1],
                    imgPadded[i, j],
                    imgPadded[i, j + 1],
                    imgPadded[i + 1, j-1],
                    imgPadded[i + 1, j],
                    imgPadded[i + 1, j + 1]]
            temp = sorted(temp)
            #print("temp ",temp)
            denoise_img[i, j]= temp[4]
            
    denoise_img = np.delete(denoise_img, (0), axis=0)
    denoise_img = np.delete(denoise_img, (denoise_img.shape[0]-1), axis=0)
    denoise_img = np.delete(denoise_img, (0), axis=1)
    denoise_img = np.delete(denoise_img, (denoise_img.shape[1]-1), axis=1)
       
    #print("filter",denoise_img.shape,img.shape)
    #print("00000000",denoise_img)
    imwrite('new_median_filtered.png', denoise_img)
    #raise NotImplementedError
    denoise_img = denoise_img.astype(np.uint8)
    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here
    kernel = np.flipud(np.fliplr(kernel))
    m,n = img.shape
    outputImg = np.zeros([m+2, n+2])
    imgPadded = np.pad(img, 1, mode='constant')

    # Loop over every pixel of the image
    for i in range(1,m-1):
        for j in range(1, n-1):
            outputImg[i, j] = (kernel * imgPadded[i: i+3, j: j+3]).sum()
            
    #print("output ",outputImg)
    outputImg = np.delete(outputImg, (0), axis=0)
    outputImg = np.delete(outputImg, (outputImg.shape[0]-1), axis=0)
    outputImg = np.delete(outputImg, (0), axis=1)
    outputImg = np.delete(outputImg, (outputImg.shape[1]-1), axis=1)

    #print("convolve 2d",outputImg.shape,img.shape)
    #raise NotImplementedError
    return outputImg


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #print("inside edge detext")
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    edge_xTemp = convolve2d(img, sobel_x)
    tempXM = (edge_xTemp - edge_xTemp.min())/(edge_xTemp.max()-edge_xTemp.min())
    edge_x = 255*tempXM
    edge_yTemp = convolve2d(img, sobel_y)
    tempYM = (edge_yTemp - edge_yTemp.min())/(edge_yTemp.max()-edge_yTemp.min())
    edge_y = 255 * tempYM
    #print("edgex ",edge_x)
    #print("edgey ",edge_y)
    imwrite('task2_edge_x.jpg', edge_x)
    imwrite('task2_edge_y.jpg', edge_y)
    #print("dtype fo x ",edge_x.dtype)

    matrixX = edge_xTemp**2
    matrixY = edge_yTemp**2
    #rows,columns = matrixX.shape
    #result = np.zeros([rows, columns])
    #for i in range(rows):
        #for j in range(columns):
            #result[i][j] = matrixX[i][j]+matrixY[i][j]
        
    edgeMagTemp = np.sqrt(matrixX + matrixY)
    tempMag = (edgeMagTemp - edgeMagTemp.min())/(edgeMagTemp.max()-edgeMagTemp.min())
    edge_mag = 255 * tempMag
    #print("edge mag ",type(edge_mag))
    #print("dtype ",edge_mag.dtype)
    edge_mag = edge_mag.astype('float64')
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    edge_mag = edge_mag.astype(np.uint8)

    imwrite('task2_edge_mag.jpg', edge_mag)
    #raise NotImplementedError
    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    kernel_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).astype(int)
    kernel_45 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]).astype(int)
    
    #print("kernel_45 ",kernel_45)
    #print("kernel_135 ",kernel_135)
    
    edge_45Temp = convolve2d(img, kernel_45)
    temp45M = (edge_45Temp - edge_45Temp.min())/(edge_45Temp.max()-edge_45Temp.min())
    edge_45 = 255*temp45M

    edge_135Temp = convolve2d(img, kernel_135)
    temp135M = (edge_135Temp - edge_135Temp.min())/(edge_135Temp.max()-edge_135Temp.min())
    edge_135 = 255*temp135M

    imwrite('task2_edge_45.jpg', edge_45)
    imwrite('task2_edge_135.jpg', edge_135)

    # print the two kernels you designed here
    edge_45 = edge_45.astype(np.uint8)
    edge_135 = edge_135.astype(np.uint8)

    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)


