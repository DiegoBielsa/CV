#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Bundle Adjustment and Multiview Geometry
#
# Date: 26 October 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio
import math as math

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
    

def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)
    
def on_click_epipolar(event):
    if event.button == 1:  # Left mouse button
        plt.figure(4)
        plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
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
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Image 1 - Click a point')
    plt.draw()  # We update the figure display
    fig.canvas.mpl_connect('button_press_event', on_click_epipolar)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return;

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

def getFundamentalMatrix(points1, points2):
    """
    -input:
        points1: np.array normalized in homogeneous 2d points in camera 1, matches of points2 (nx2)
        points2: np.array normalized 2d points in camera 1, matches of points1 (nx2)
    -output:
        F_21: fundamental matrix F_21
    """
    x1Data = points1.T
    x2Data = points2.T
    
    A = []

    for i in range(x1Data.shape[1]):
        x0, y0, w0 = x1Data[:, i]
        x1, y1, w1 = x2Data[:, i]
        A.append([x0*x1, y0*x1, w0*x1, x0*y1, y0*y1, w0*y1, x0*w1, y0*w1, w0*w1])
        
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    F_c2_c1_estimated = V[-1].reshape(3, 3)
    rank = np.linalg.matrix_rank(F_c2_c1_estimated)
    U, S, V = np.linalg.svd(F_c2_c1_estimated)
    S[2:]=0
    F_c2_c1_estimated = np.dot(U,np.dot(np.diag(S),V))
    rank = np.linalg.matrix_rank(F_c2_c1_estimated)
    
    return F_c2_c1_estimated / F_c2_c1_estimated[2][2]

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]], dtype="object")
    return M

#theta = crossMatrixInv(sc.linalg.logm(R))

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    -input:
        Op: Optimization parameters: this must include a
            paramtrization for T_21 (reference 1 seen from reference 2) Op = [theta_transl(incl), phi (athimuth), theta1, theta2, theta3, x, y, z, w]
            in a proper way and for X1 (3D points in ref 1, our 3d points)             
        x1Data: (3xnPoints) 2D points on image 1 (homogeneous
            coordinates) [[x], [y], [w]]
        x2Data: (3xnPoints) 2D points on image 2 (homogeneous
            coordinates) [[x], [y], [w]]
        K_c: (3x3) Intrinsic calibration matrix
        nPoints: Number of points
    -output:
        res: residuals from the error between the 2D matched points
            and the projected points from the 3D points
            (2 equations/residuals per 2D point)
    """
    ######################### Get params #########################
    theta = np.array([Op[2], Op[3], Op[4]])
    R_c2_c1 = sc.linalg.expm(crossMatrix(theta))
    t_c2_c1 = np.array([np.sin(Op[0]) * np.cos(Op[1]), np.sin(Op[0]) * np.sin(Op[1]), np.cos(Op[0])])
    T_c2_c1 = np.vstack((np.hstack((R_c2_c1, t_c2_c1[:, np.newaxis])), [0, 0, 0, 1]))
    
    P_canonical = np.array([[1, 0, 0 ,0], [0, 1, 0 ,0], [0, 0, 1, 0]]);
    P_c1 = K_c @ P_canonical;
    P_c2 = K_c @ P_canonical @ T_c2_c1;
    
    ######################### Get 3D points in cam 1 and 2 #########################
    p3D_1 = []
    for i in range(0, nPoints * 4, 4):
        x = Op[i + 5]
        y = Op[i + 6]
        z = Op[i + 7]
        w = Op[i + 8]
        p3D_1.append(np.array([x,y,x,w]))
    p3D_1 = p3D_1.T;
    
    p3D_2 = []
    for i in range(p3D_1.shape[1]):
        p3D_2.append(T_c2_c1 @ p3D_1[:, i])
    p3D_2 = np.array(p3D_2);
    
    ######################### Project 3d points to each camera #########################
    p2D_1 = [];
    p2D_2 = [];
    for i in range(p3D_1.shape[1]):
        p2D_1.append(P_c1 @ p3D_1[:, i])
        p2D_2.append(P_c2 @ p3D_2[:, i])
    p2D_1 = np.array(p2D_1);
    p2D_2 = np.array(p2D_2);
    
    loss = [];
    for i in range(nPoints):
        e_1x = x1Data[0, i] - p2D_1[0, i];
        e_1y = x1Data[1, i] - p2D_1[1, i];
        e_2x = x2Data[0, i] - p2D_2[0, i];
        e_2y = x2Data[1, i] - p2D_2[1, i];
        loss.append(e_1x);
        loss.append(e_1y);
        loss.append(e_2x);
        loss.append(e_2y);
    #loss = np.array(loss);
    return loss;

def distanceLinePoint(line, point):
    return abs(line[0] * point[0] + line[1] * point[1] + line[2]) / math.sqrt(line[0] ** 2 + line[1] ** 2)

def euclideanDistance2D(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def euclideanDistance3d(point1, point2):
    
    dehomogenized_point1 = point1 / point1[3]
    dehomogenized_point2 = point2 / point2[3]

    distance = np.linalg.norm(dehomogenized_point1 - dehomogenized_point2)

    return distance


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_wc1 = np.loadtxt('T_w_c1.txt')
    T_wc2 = np.loadtxt('T_w_c2.txt')                                    
    T_wc3 = np.loadtxt('T_w_c3.txt')
    F_21_ground_truth = np.loadtxt('F_21.txt')
    K_c = np.loadtxt('K_c.txt')
    X_w = np.loadtxt('X_w.txt')

    x1Data = np.loadtxt('x1Data.txt')
    x2Data = np.loadtxt('x2Data.txt')
    x3Data = np.loadtxt('x3Data.txt')


    #Plot the 3D cameras and the 3D points
    fig3D = plt.figure(1)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2')
    drawRefSystem(ax, T_wc3, '-', 'C3')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', 0.1)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    #plt.show()


    #Read the images
    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'
    path_image_3 = 'image3.png'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)
    
    N1 = normalizationMatrix(image_pers_1.shape[1], image_pers_1.shape[0])
    N2 = normalizationMatrix(image_pers_2.shape[1], image_pers_2.shape[0])


    # Construct the matches
    kpCv1 = []
    kpCv2 = []
    kpCv3 = []
    for kPoint in range(x1Data.shape[1]):
        kpCv1.append(cv2.KeyPoint(x1Data[0, kPoint], x1Data[1, kPoint],1))
        kpCv2.append(cv2.KeyPoint(x2Data[0, kPoint], x2Data[1, kPoint],1))
        kpCv3.append(cv2.KeyPoint(x3Data[0, kPoint], x3Data[1, kPoint],1))

    matchesList12 = np.hstack((np.reshape(np.arange(0, x1Data.shape[1]),(x2Data.shape[1],1)),
                                        np.reshape(np.arange(0, x1Data.shape[1]), (x1Data.shape[1], 1)),np.ones((x1Data.shape[1],1))))

    matchesList13 = matchesList12
    dMatchesList12 = indexMatrixToMatchesList(matchesList12)
    dMatchesList13 = indexMatrixToMatchesList(matchesList13)

    imgMatched12 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_2, kpCv2, dMatchesList12,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    imgMatched13 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_3, kpCv3, dMatchesList13,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(2)
    plt.imshow(imgMatched12)
    plt.title("{} matches between views 1 and 2".format(len(dMatchesList12)))
    #plt.draw()

    plt.figure(3)
    plt.imshow(imgMatched13)
    plt.title("{} matches between views 1 and 3".format(len(dMatchesList13)))
    print('Close the figures to continue.')
    #plt.show()

    # Project the points
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ X_w
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3) @ X_w
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]


    # Plot the 2D points
    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p, 'k-')
    plt.plot(x1_p[0, :], x1_p[1, :], 'bo')
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.title('Image 1')
    #plt.draw()

    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p, 'k-')
    plt.plot(x2_p[0, :], x2_p[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plotNumberedImagePoints(x2Data[0:2, :], 'r', 4)
    plt.title('Image 2')
    #plt.draw()

    plt.figure(6)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_p, 'k-')
    plt.plot(x3_p[0, :], x3_p[1, :], 'bo')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plotNumberedImagePoints(x3Data[0:2, :], 'r', 4)
    plt.title('Image 3')
    print('Close the figures to continue.')
    #plt.show()
    
    # ----------------------------- USING OUR F -----------------------------
    p1_norm = [];
    x1Data_T = np.vstack((x1Data, np.ones((1, x1Data.shape[1])))).T
    for i in range(x1Data_T.shape[0]):
        x1Norm = N1 @ np.array([x1Data_T[i][0], x1Data_T[i][1], x1Data_T[i][2]]).T;
        p1_norm.append(x1Norm);
    p1_norm = np.array(p1_norm);
    
    p2_norm = [];
    x2Data_T = np.vstack((x2Data, np.ones((1, x2Data.shape[1])))).T
    for i in range(x2Data_T.shape[0]):
        x2Norm = N2 @ np.array([x2Data_T[i][0], x2Data_T[i][1], x1Data_T[i][2]]).T;
        p2_norm.append(x2Norm);
    p2_norm = np.array(p2_norm);
    
    F_21 = getFundamentalMatrix(x1Data_T, x2Data_T);
    #For unnormalizing the resulting F matrix, before evaluating the matches:
    print(F_21)
    print(F_21_ground_truth)
    drawEpipolarLine(21);
    T_c1_w = np.linalg.inv(T_wc1);
    T_c2_w = np.linalg.inv(T_wc2);
    # Canonical perspective projection matrix
    P_canonical = np.array([[1, 0, 0 ,0], [0, 1, 0 ,0], [0, 0, 1, 0]]);
    
    # Projection matrixes
    P_c1 = K_c @ P_canonical @ T_c1_w;
    P_c2 = K_c @ P_canonical @ T_c2_w;
    
    
    E_c2_c1_estimated = (K_c.T) @ F_21 @ K_c
    
    U, _, V = np.linalg.svd(E_c2_c1_estimated)
    
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ V # calcular determinante
    np.linalg.det(R1);
    if np.round(np.linalg.det(R1)) == -1:
        R1 *= -1
    R2 = U @ W.T @ V # calculo determinante y si sale -1 la multiplico por -1
    np.linalg.det(R2);
    if np.round(np.linalg.det(R2)) == -1:
        R2 *= -1
    
    
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    T_c2_c1_estimated0 = np.vstack((np.hstack((R1, t1[:, np.newaxis])), [0, 0, 0, 1]))
    T_c2_c1_estimated1 = np.vstack((np.hstack((R1, t2[:, np.newaxis])), [0, 0, 0, 1]))
    T_c2_c1_estimated2 = np.vstack((np.hstack((R2, t1[:, np.newaxis])), [0, 0, 0, 1]))
    T_c2_c1_estimated3 = np.vstack((np.hstack((R2, t2[:, np.newaxis])), [0, 0, 0, 1]))
    
    # Triangulation of one point (The first one for example)
    
    x0, y0 = x1Data[:, 0]
    x1, y1 = x2Data[:, 0]
    A = np.zeros((4, 4))
    A[0] = x0 * P_c1[2] - P_c1[0]
    A[1] = y0 * P_c1[2] - P_c1[1]
    A[2] = x1 * P_c2[2] - P_c2[0]
    A[3] = y1 * P_c2[2] - P_c2[1]
    
    _, _, V = np.linalg.svd(A)
    X_homogeneous = V[-1, :]
    X = X_homogeneous / X_homogeneous[3]
    
    # Transform the points to the camera frames
    X_c1 = T_c1_w @ X
    X_c2 = T_c2_w @ X
    
    
    # I transform 
    X_c2_estimated0 = T_c2_c1_estimated0 @ X_c1
    d0 = euclideanDistance3d(X_c2_estimated0, X_c2) 
    X_c2_estimated1 = T_c2_c1_estimated1 @ X_c1
    d1 = euclideanDistance3d(X_c2_estimated1, X_c2)
    X_c2_estimated2 = T_c2_c1_estimated2 @ X_c1 
    d2 = euclideanDistance3d(X_c2_estimated2, X_c2) # this is the one, less euclidean distance
    X_c2_estimated3 = T_c2_c1_estimated3 @ X_c1
    d3 = euclideanDistance3d(X_c2_estimated3, X_c2)
    
    fig3D = plt.figure(6)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    T_c2_c1 = T_c2_w @ T_wc1;
    drawRefSystem(ax, T_c2_c1_estimated2, '-', 'C2_estimated')
    drawRefSystem(ax, T_c2_c1, '-', 'C2')


    ax.scatter([X_c2[0], X_c2_estimated0[0]], [X_c2[1], X_c2_estimated0[1]], [X_c2[2], X_c2_estimated0[2]], marker='.')

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()
    a = 0
