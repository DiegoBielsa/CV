import matplotlib.pyplot as plt
import numpy as np
import cv2 
import random
import math

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
    
    img1 = cv2.cvtColor(cv2.imread('labSession3/image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('labSession3/image2.png'), cv2.COLOR_BGR2RGB)

    
    superglue_matches_path = 'SuperGlueResultsLab3/image1_image2_matches.npz'
    npz = np.load(superglue_matches_path)
    keypoints0 = npz['keypoints0']
    keypoints1 = npz['keypoints1']
    # Matches is an array of keyponts0.shape[0] (number of points). Each component contains the index of the match in keypoints1
    matches = npz['matches']
    
    x1 = [];
    x2 = [];

    for i in range(matches.size):
        if matches[i] != -1: # here there is a match
            match_x_0 = keypoints0[i];
            x1.append(match_x_0);
            match_x_1 = keypoints1[matches[i]];
            x2.append(match_x_1);
            
    x1 = np.array(x1);
    x2 = np.array(x2);
    H_21 = getHomographyMatrix(x1, x2)
    
    #drawHomography(21);
    
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
    print(npz['matches'])