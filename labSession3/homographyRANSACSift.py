#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: SIFT matching
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2 
import random
import math

def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        NNDR=dist[indexSort[0]] / dist[indexSort[1]]
        if ((dist[indexSort[0]] < minDist) & (NNDR < distRatio)):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])


    return matches

def on_click_homography(event):
    if event.button == 1:  # Left mouse button
        plt.figure(34)
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.title('Image 2 - Homography')
        plt.draw()  # We update the figure display
        print(f'You clicked at ({event.xdata}, {event.ydata})')
        p1 = np.array([event.xdata, event.ydata, 1])
        p2 = np.dot(H_21, p1) # apply homography
        p2 /= p2[2]
        plt.plot(p2[0], p2[1],'rx', markersize=10)

def drawHomography (figNum): # Draw epipolar line of a clicked point
    fig = plt.figure(figNum)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.title('Image 1 - Click a point on the ground plane')
    plt.draw()  # We update the figure display
    fig.canvas.mpl_connect('button_press_event', on_click_homography)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return;

def getHomographyMatrix(points1, points2):

    x1, y1 = points1.T
    x2, y2 = points2.T
    
    A = []
    for i in range(len(x1)):
        A.append([x1[i], y1[i], 1, 0, 0, 0, -x2[i]*x1[i], -x2[i]*y1[i], -x2[i]])
        A.append([0, 0, 0, x1[i], y1[i], 1, -y2[i]*x1[i], -y2[i]*y1[i], -y2[i]])
    A = np.array(A)
    #print(A.shape)
    
    u, s, vh = np.linalg.svd(A);
    H_21_estimated = vh[-1].reshape(3, 3)
    
    H_21_estimated = H_21_estimated / H_21_estimated[2][2]
    return H_21_estimated

def euclideanDistance2D(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

if __name__ == '__main__':
    
    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    distRatio = 0.8
    minDist = 200
    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Plot the first 10 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    #x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    #x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))
    x1 = srcPts;
    x2 = dstPts;

#################################### RANSAC ####################################
    # parameters of random sample selection
    """spFrac = nOutliers/nInliers  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 4  # we need 4 matches at least to compute the H matrix
    thresholdFactor = 1.96  # a point is spurious if abs(r/s)>factor Threshold

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)
    print('nAttempts = ' + str(nAttempts))

    nElements = x.shape[1]

    RANSACThreshold = 3*inliersSigma
    nVotesMax = 0
    rng = np.random.default_rng()"""
        
    RANSACThreshold = 12
    nVotesMax = 0
    votesMax = [False] * x1.shape[0];
    pMinSet = 4

    for kAttempt in range(100):
    
        # Compute the minimal set defining your model
        i0 = random.randint(0, x1.shape[0] - 1)
        i1 = random.randint(0, x1.shape[0] - 1)
        i2 = random.randint(0, x1.shape[0] - 1)
        i3 = random.randint(0, x1.shape[0] - 1)
        
        p1 = [];
        p1.append(x1[i0]);
        p1.append(x1[i1]);
        p1.append(x1[i2]);
        p1.append(x1[i3]);
        p1 = np.array(p1);
        
        p2 = [];
        p2.append(x2[i0]);
        p2.append(x2[i1]);
        p2.append(x2[i2]);
        p2.append(x2[i3]);
        p2 = np.array(p2);
        
        
        H_21_estimated = getHomographyMatrix(p1, p2);
        votes = [False] * x1.shape[0];
        nVotes = 0;

        # Now we calculate the votes regarding the euclidean distance between the matched point and the input one (assuming it is ok)
        for i in range(x1.shape[0]):
            p1_to_estimate = np.array([x1[i][0], x1[i][1], 1]);
            p2_estimated = np.dot(H_21_estimated, p1_to_estimate);
            p2_estimated /= p2_estimated[2]
            p2_dehomogenized = np.array([p2_estimated[0], p2_estimated[1]]);
            euclidean_dist = euclideanDistance2D(p2_dehomogenized, x2[i]);
            if euclidean_dist < RANSACThreshold:
                votes[i] = True;
                nVotes += 1;

        if nVotes > nVotesMax:
            nVotesMax = nVotes
            votesMax = votes
            H_21_most_voted = H_21_estimated
           
    H_21 = H_21_most_voted;
    drawHomography(21);