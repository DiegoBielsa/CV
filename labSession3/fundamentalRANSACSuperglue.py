import matplotlib.pyplot as plt
import numpy as np
import cv2 
import random
import math

F_21_ground_truth = np.loadtxt('F_21_test.txt')

def normalizationMatrix(nx,ny):
    """
 
    -input:
        nx: number of columns of the matrix
        ny: number of rows of the matrix
    -output:
        Nv: normalization matrix such that xN = Nv @ x
    """
    Nv = np.array([[1/nx, 0, -1/2], [0, 1/ny, -1/2], [0, 0, 1]])
    return Nv
 

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


def on_click_epipolar(event):
    if event.button == 1:  # Left mouse button
        plt.figure(4)
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.title('Image 2 - Epipolar Lines')
        plt.draw()  # We update the figure display
        print(f'You clicked at ({event.xdata}, {event.ydata})')
        x_0 = np.array([event.xdata, event.ydata, 1])
        l_xi_1 = np.dot(F_21, x_0);
        u, s, vh = np.linalg.svd(F_21_ground_truth.T);
        e_2 = vh[-1, :];
        e_2 = e_2 / e_2[2]
        plt.plot(e_2[0], e_2[1],'rx', markersize=10)
        u, s, vh = np.linalg.svd(F_21.T);
        e_2 = vh[-1, :];
        e_2 = e_2 / e_2[2]
        plt.plot(e_2[0], e_2[1],'bx', markersize=10)
        drawLine(l_xi_1, 'g-', 1)
                

def drawEpipolarLine (figNum): # Draw epipolar line of a clicked point
    fig = plt.figure(figNum)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.title('Image 1 - Click a point')
    plt.draw()  # We update the figure display
    fig.canvas.mpl_connect('button_press_event', on_click_epipolar)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return;

def getFundamentalMatrix(points1, points2):
    x1Data = points1.T
    x2Data = points2.T
    
    A = []

    for i in range(x1Data.shape[1]):
        x0, y0 = x1Data[:, i]
        x1, y1 = x2Data[:, i]
        A.append([x0*x1, y0*x1, x1, x0*y1, y0*y1, y1, x0, y0, 1])
        
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    F_c2_c1_estimated = V[-1].reshape(3, 3)
    rank = np.linalg.matrix_rank(F_c2_c1_estimated)
    U, S, V = np.linalg.svd(F_c2_c1_estimated)
    S[2:]=0
    F_c2_c1_estimated = np.dot(U,np.dot(np.diag(S),V))
    rank = np.linalg.matrix_rank(F_c2_c1_estimated)
    
    return F_c2_c1_estimated

def distanceLinePoint(line, point):
    return abs(line[0] * point[0] + line[1] * point[1] + line[2]) / math.sqrt(line[0] ** 2 + line[1] ** 2)

def euclideanDistance2D(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

if __name__ == '__main__':
    
    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
    
    N1 = normalizationMatrix(img1.shape[1], img1.shape[0])
    N2 = normalizationMatrix(img2.shape[1], img2.shape[0])
    
    
    img_matches = cv2.cvtColor(cv2.imread('../SuperGlueResultsLab3/image1_image2_matches.png'), cv2.COLOR_BGR2RGB)
    plt.imshow(img_matches, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()

    
    superglue_matches_path = '../SuperGlueResultsLab3/image1_image2_matches.npz'
    npz = np.load(superglue_matches_path)
    keypoints0 = npz['keypoints0']
    keypoints1 = npz['keypoints1']
    # Matches is an array of keyponts0.shape[0] (number of points). Each component contains the index of the match in keypoints1
    matches = npz['matches']
    
    x1 = [];
    x2 = [];
    matches_matrix = [];

    for i in range(matches.size):
        if matches[i] != -1: # here there is a match
            match_x_0 = keypoints0[i];
            x1.append(match_x_0);
            match_x_1 = keypoints1[matches[i]];
            x2.append(match_x_1);
            matches_matrix.append([i, matches[i]])
    
    # We use only the matches on the floor, in order to compute the omography with respect to the floor plane
    x1 = np.array(x1);
    x2 = np.array(x2);
    matches_matrix = np.array(matches_matrix);
    
    
    
    
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
    pMinSet = 8
    p1_selected = []
    p2_selected = []
    F_21_most_voted = []
    

    for kAttempt in range(100):
    
        # Compute the minimal set defining your model
        i0 = random.randint(0, x1.shape[0] - 1)
        i1 = random.randint(0, x1.shape[0] - 1)
        i2 = random.randint(0, x1.shape[0] - 1)
        i3 = random.randint(0, x1.shape[0] - 1)
        i4 = random.randint(0, x1.shape[0] - 1)
        i5 = random.randint(0, x1.shape[0] - 1)
        i6 = random.randint(0, x1.shape[0] - 1)
        i7 = random.randint(0, x1.shape[0] - 1)
        
        # Normalized to compute F norm
        p1_norm = [];
        x1Norm = N1 @ np.array([x1[i0][0], x1[i0][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i1][0], x1[i1][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i2][0], x1[i2][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i3][0], x1[i3][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i4][0], x1[i4][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i5][0], x1[i5][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i6][0], x1[i6][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        x1Norm = N1 @ np.array([x1[i7][0], x1[i7][1], 1]);
        p1_norm.append([x1Norm[0], x1Norm[1]]);
        p1_norm = np.array(p1_norm);
        
        p2_norm = [];
        x2Norm = N2 @ np.array([x2[i0][0], x2[i0][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i1][0], x2[i1][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i2][0], x2[i2][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i3][0], x2[i3][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i4][0], x2[i4][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i5][0], x2[i5][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i6][0], x2[i6][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        x2Norm = N2 @ np.array([x2[i7][0], x2[i7][1], 1]);
        p2_norm.append([x2Norm[0], x2Norm[1]]);
        p2_norm = np.array(p2_norm);
        
        # Not normalized to plot
        p1 = [];
        p1.append(x1[i0]);
        p1.append(x1[i1]);
        p1.append(x1[i2]);
        p1.append(x1[i3]);
        p1.append(x1[i4]);
        p1.append(x1[i5]);
        p1.append(x1[i6]);
        p1.append(x1[i7]);
        p1 = np.array(p1);
        
        p2 = [];
        p2.append(x2[i0]);
        p2.append(x2[i1]);
        p2.append(x2[i2]);
        p2.append(x2[i3]);
        p2.append(x2[i4]);
        p2.append(x2[i5]);
        p2.append(x2[i6]);
        p2.append(x2[i7]);
        p2 = np.array(p2);
       
        
        if kAttempt % 10 == 0:
            # Each 10 iterations, the hypotesis is going to be shown
            result_img_local = np.concatenate((img1, img2), axis=1)
            for j in range(pMinSet):
                x1_local = p1[j][0]
                y1_local = p1[j][1]
                x2_local = p2[j][0] + img1.shape[1]
                y2_local = p2[j][1]
                cv2.line(result_img_local, (int(x1_local), int(y1_local)), (int(x2_local), int(y2_local)), (0, 255, 0), 3)  # Draw a line between matches
            plt.imshow(result_img_local, cmap='gray', vmin=0, vmax=255)
            plt.draw()
            plt.waitforbuttonpress()  
        
        
        Fn_21_estimated = getFundamentalMatrix(p1_norm, p2_norm);
        #For unnormalizing the resulting F matrix, before evaluating the matches:
        F_21_estimated = N2.T @ Fn_21_estimated @ N1
        votes = [False] * x1.shape[0];
        nVotes = 0;

        # Now we calculate the votes regarding the euclidean distance between the matched point and the input one (assuming it is ok)
        for i in range(x1.shape[0]):
            p1_to_estimate = np.array([x1[i][0], x1[i][1], 1]);
            l2_estimated = np.dot(F_21_estimated, p1_to_estimate);
            distance = distanceLinePoint(l2_estimated, x2[i])
            if distance < RANSACThreshold:
                votes[i] = True;
                nVotes += 1;

        if nVotes > nVotesMax:
            nVotesMax = nVotes
            votesMax = votes
            F_21_most_voted = F_21_estimated
            p1_selected = p1
            p2_selected = p2
           
    F_21 = F_21_most_voted;
    result_img_final = np.concatenate((img1, img2), axis=1)
    for j in range(pMinSet):
        x1_final = p1[j][0]
        y1_final = p1[j][1]
        x2_final = p2[j][0] + img1.shape[1]
        y2_final = p2[j][1]
        cv2.line(result_img_final, (int(x1_final), int(y1_final)), (int(x2_final), int(y2_final)), (255, 0, 0), 3)  # Draw a line between matches
    plt.imshow(result_img_final, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress() 
    
    u, s, vh = np.linalg.svd(F_21_ground_truth.T);
    e_ground_truth = vh[-1, :];
    e_ground_truth = e_ground_truth / e_ground_truth[2]
    u, s, vh = np.linalg.svd(F_21.T);
    e_2 = vh[-1, :];
    e_2 = e_2 / e_2[2]
    print("Ground truth epipole:")
    print(e_ground_truth)
    print("Computed epipole:")
    print(e_ground_truth)
    print("Epipole error:")
    print(euclideanDistance2D(e_ground_truth, e_2));
    
        
    drawEpipolarLine(21);
