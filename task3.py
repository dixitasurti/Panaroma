#!/usr/bin/env python
# coding: utf-8

# In[19]:


"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np


def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    structuringElements= np.ones((3,3), dtype=np.uint8)
    m,n= img.shape
    constant= 1
    erode_img= np.zeros((m,n), dtype=np.uint8)
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*structuringElements
            erode_img[i,j]= np.min(product)
            
    #raise NotImplementedError
    erode_img = erode_img.astype(np.uint8)
    return erode_img


def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    structuringElements= np.ones((3,3), dtype=np.uint8)
    m,n= img.shape
    constant= 1
    dilate_img= np.zeros((m,n), dtype=np.uint8)
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*structuringElements
            dilate_img[i,j]= np.max(product)
            
    #raise NotImplementedError
    dilate_img = dilate_img.astype(np.uint8)
    return dilate_img


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    #Erosion + dilation
    erodedImg = morph_erode(img)
    open_img = morph_dilate(erodedImg)
    #raise NotImplementedError
    open_img = open_img.astype(np.uint8)
    return open_img


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    #dilation + erosion
    dilatedImg = morph_dilate(img)
    close_img = morph_erode(dilatedImg)
    #raise NotImplementedError
    close_img = close_img.astype(np.uint8)
    return close_img


def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    
    # TO DO: implement your solution here
    #print("going for eorde")
    imgErodePadded = np.pad(img, 1, mode='constant')
    erodedImg = morph_erode(imgErodePadded)
    
    erodedImg = np.delete(erodedImg, (0), axis=0)
    erodedImg = np.delete(erodedImg, (erodedImg.shape[0]-1), axis=0)
    erodedImg = np.delete(erodedImg, (0), axis=1)
    erode_img = np.delete(erodedImg, (erodedImg.shape[1]-1), axis=1)
    #print("shape of eroded image ",erode_img.shape,img.shape)
    imwrite("Eroded.png", erode_img)
    #print("done with eorde")
    
    
    #print("going for dilate")
    imgDilatedPadded = np.pad(img, 1, mode='constant')
    dilatedImg = morph_dilate(imgDilatedPadded)
    dilatedImg = np.delete(dilatedImg, (0), axis=0)
    dilatedImg = np.delete(dilatedImg, (dilatedImg.shape[0]-1), axis=0)
    dilatedImg = np.delete(dilatedImg, (0), axis=1)
    dilate_img = np.delete(dilatedImg, (dilatedImg.shape[1]-1), axis=1)
    #print("shape of dilate_img image ",dilate_img.shape,img.shape)
    imwrite("Dilated.png", dilate_img)
    #print("done with dilate")
    
    
    #print("going for open")
    imgPadded = np.pad(img, 1, mode='constant')
    openedImg = morph_open(imgPadded)
    openedImgTemp = openedImg
    openedImg = np.delete(openedImg, (0), axis=0)
    openedImg = np.delete(openedImg, (openedImg.shape[0]-1), axis=0)
    openedImg = np.delete(openedImg, (0), axis=1)
    open_img = np.delete(openedImg, (openedImg.shape[1]-1), axis=1)
    #print("shape of open_img image ",open_img.shape,img.shape)
    imwrite("opened.png", open_img)  
    #print("done with ooen")
    
    
    #print("going for close")
    imgPadded = np.pad(img, 1, mode='constant')
    closedImg = morph_close(imgPadded)
    closedImg = np.delete(closedImg, (0), axis=0)
    closedImg = np.delete(closedImg, (closedImg.shape[0]-1), axis=0)
    closedImg = np.delete(closedImg, (0), axis=1)
    close_img = np.delete(closedImg, (closedImg.shape[1]-1), axis=1)
    #print("shape of close_img image ",close_img.shape,img.shape)
    imwrite("closed.png", close_img)  
    #print("done with close")
    
    
    #for denoising = opening + closing or vice versa
    #print("started denoising")
    denoise_img = morph_close(openedImgTemp)
    denoise_img = np.delete(denoise_img, (0), axis=0)
    denoise_img = np.delete(denoise_img, (denoise_img.shape[0]-1), axis=0)
    denoise_img = np.delete(denoise_img, (0), axis=1)
    denoise_img = np.delete(denoise_img, (denoise_img.shape[1]-1), axis=1)
    #print("shape of task3_denoise image ",denoise_img.shape,img.shape)
    imwrite("task3_denoise.png", denoise_img) 
    #print("done with denoising")
    
    #raise NotImplementedError
    denoise_img = denoise_img.astype(np.uint8)
    return denoise_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    print("Started boundary")
    imgPadded = np.pad(img, 1, mode='constant')
    erodedImg = morph_erode(imgPadded)
    bound_img = imgPadded - erodedImg
    
    bound_img = np.delete(bound_img, (0), axis=0)
    bound_img = np.delete(bound_img, (bound_img.shape[0]-1), axis=0)
    bound_img = np.delete(bound_img, (0), axis=1)
    bound_img = np.delete(bound_img, (bound_img.shape[1]-1), axis=1)

    imwrite("bound_img.png", bound_img) 
    #print("shape of bound_img image ",bound_img.shape,img.shape)
    #print("donw iwth boundary")
    #raise NotImplementedError
    bound_img = bound_img.astype(np.uint8)
    return bound_img


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)
