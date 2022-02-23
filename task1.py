#!/usr/bin/env python
# coding: utf-8

# In[53]:


"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random
 
def geometricDistance(goodMatch, h_vector):
    #print("good match geometri c ",goodMatch)
    pointA = np.transpose(np.matrix([goodMatch[0][0], goodMatch[0][1], 1]))
    estimated = np.dot(h_vector, pointA)
    estimated = (1/estimated.item(2))*estimated

    pointB = np.transpose(np.matrix([goodMatch[1][0], goodMatch[1][1], 1]))
    error = pointB - estimated
    return np.linalg.norm(error)


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kpsL, descL) = descriptor.detectAndCompute(left_img, None)
    (kpsR, descR) = descriptor.detectAndCompute(right_img, None)
    print("length ",len(kpsL))
    print("keypoints",kpsL[0].pt)
    neighborsList = list()
    for i in range(len(descL)):
        #print("start ----")
        left = np.array(descL[i])
        right = np.array(descR)
        dist = np.linalg.norm(np.subtract(left,right),axis=1)
        rightPoints =[]
        for k in range(len(kpsR)):
            rightPoints.append(kpsR[k].pt)
        distances = list()
        for j in range(len(dist)):
            distances.append([rightPoints[j],dist[j],descR[j]])
        distances.sort(key=lambda x:x[1])
        neighbors = list()
        for a in range(2):
            neighbors.append((kpsL[i].pt,distances[a][0],left,distances[a][2]))
        neighborsList.append(neighbors)
    good = []
    for n in range(len(neighborsList)):
        distM = np.linalg.norm(np.subtract(neighborsList[n][0][2],neighborsList[n][0][3]))
        distN = np.linalg.norm(np.subtract(neighborsList[n][1][2],neighborsList[n][1][3]))
        if distM <0.5*distN :
            good.append(neighborsList[n][0])
    #print("good matches -- ",good)
    maxInliers = []
    threshold = 5
    finalH = None
    #pts1 = [item[0] for item in good_knns]
    #pts2 = [item[1] for item in good_knns]
    # H, mask = cv2.findHomography(np.asarray(pts1), np.asarray(pts2), cv2.RANSAC)
    # print('Homography Matrix')
    # print(H)
    # finalH = np.linalg.inv(H)
    
    
    for k in range(5000):
        randomPoints = []
        point1 = good[random.randrange(0, len(good))]
        randomPoints.append(point1)
        point2 = good[random.randrange(0, len(good))]
        randomPoints.append(point2)
        point3 = good[random.randrange(0, len(good))]
        randomPoints.append(point3)
        point4 = good[random.randrange(0, len(good))]
        randomPoints.append(point4)
        #print("randomPoints ",randomPoints)
        A = np.zeros((2*len(randomPoints),9), dtype=np.float32)
        for i in range(len(randomPoints)):
            left = randomPoints[i][0]
            right = randomPoints[i][1]
            row_1 = np.array([ left[0], left[1], 1, 0, 0, 0, -right[0]*left[0], -right[0]*left[1], -right[0]])
            row_2 = np.array([ 0, 0, 0, left[0], left[1], 1, -right[1]*left[0], -right[1]*left[1], -right[1]])
            A[2*i] = row_1
            A[(2*i)+1] = row_2
            u, s, vt = np.linalg.svd(A)  
            h_vector = vt[-1]
            h_vector = h_vector.reshape(3,3)
            #h_vector = np.linalg.norm(h_vector)
            h_vector = (1/h_vector.item(8)) * h_vector
        inliers = []
        for i in range(len(good)):
            distance = geometricDistance(good[i], h_vector)
            if distance < 5:
                inliers.append(good[i])
        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h_vector
            
        if len(maxInliers) > (len(good)*threshold):
            break
    
    print("final local h matrix ",finalH)
    rows1, cols1 = left_img.shape[:2]
    print("rows1 ",rows1,cols1)
    rows2, cols2 = right_img.shape[:2]
    list1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temPoints = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    
    list2 = cv2.perspectiveTransform(temPoints, finalH)
    listP= np.concatenate((list1,list2), axis=0)
    [xMin, yMin] = np.int32(listP.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(listP.max(axis=0).ravel() + 0.5)
    translation = [-xMin,-yMin]
    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    result_img = cv2.warpPerspective(left_img, H_translation.dot(finalH), (xMax-xMin, yMax-yMin))
    result_img[translation[1]:rows1+translation[1], translation[0]:cols1+translation[0]] = right_img
    cv2.imwrite("panaroma.jpg", result_img)
    result_img = result_img.astype(np.uint8)
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)




