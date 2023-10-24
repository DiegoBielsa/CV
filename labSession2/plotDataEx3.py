#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Homography, Fundamental Matrix and Two View SfM
#
# Date: 16 September 2022
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

F_21 = np.loadtxt('F_21_test.txt')
H_c2_c1_toPlot = [];

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

# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4), dtype=np.float32)
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


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
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

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
    

def on_click_homography(event):
    if event.button == 1:  # Left mouse button
        plt.figure(34)
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.title('Image 2 - Homography')
        plt.draw()  # We update the figure display
        print(f'You clicked at ({event.xdata}, {event.ydata})')
        p1 = np.array([event.xdata, event.ydata, 1])
        p2 = np.dot(H_c2_c1, p1) # apply homography
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
    

def euclideanDistance3d(point1, point2):
    
    dehomogenized_point1 = point1 / point1[3]
    dehomogenized_point2 = point2 / point2[3]

    distance = np.linalg.norm(dehomogenized_point1 - dehomogenized_point2)

    return distance


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')
    
    T_c1_w = np.linalg.inv(T_w_c1);
    T_c2_w = np.linalg.inv(T_w_c2);

    K_c = np.loadtxt('K_c.txt')
    X_w = np.loadtxt('X_w.txt')
    
    # Canonical perspective projection matrix
    P_canonical = np.array([[1, 0, 0 ,0], [0, 1, 0 ,0], [0, 0, 1, 0]]);
    
    # Projection matrixes
    P_c1 = K_c @ P_canonical @ T_c1_w;
    P_c2 = K_c @ P_canonical @ T_c2_w;

    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')
        
    A = np.ones([4,4]);
    
    rows, columns = x1.shape;
    X_own_w = np.ones([columns, 4]);
    
    for i in range(columns):
        A[0][0] = P_c1[2][0] * x1[0][i] - P_c1[0][0];
        A[0][1] = P_c1[2][1] * x1[0][i] - P_c1[0][1];
        A[0][2] = P_c1[2][2] * x1[0][i] - P_c1[0][2];
        A[0][3] = P_c1[2][3] * x1[0][i] - P_c1[0][3];
        
        A[1][0] = P_c1[2][0] * x1[1][i] - P_c1[1][0];
        A[1][1] = P_c1[2][1] * x1[1][i] - P_c1[1][1];
        A[1][2] = P_c1[2][2] * x1[1][i] - P_c1[1][2];
        A[1][3] = P_c1[2][3] * x1[1][i] - P_c1[1][3];
        
        A[2][0] = P_c2[2][0] * x2[0][i] - P_c2[0][0];
        A[2][1] = P_c2[2][1] * x2[0][i] - P_c2[0][1];
        A[2][2] = P_c2[2][2] * x2[0][i] - P_c2[0][2];
        A[2][3] = P_c2[2][3] * x2[0][i] - P_c2[0][3];
        
        A[3][0] = P_c2[2][0] * x2[1][i] - P_c2[1][0];
        A[3][1] = P_c2[2][1] * x2[1][i] - P_c2[1][1];
        A[3][2] = P_c2[2][2] * x2[1][i] - P_c2[1][2];
        A[3][3] = P_c2[2][3] * x2[1][i] - P_c2[1][3];
        
        u, s, vh = np.linalg.svd(A);
        point = vh[-1, :];
        point_n = point / point[3];
        X_own_w[i, :] = point_n;

    X_own_w = X_own_w.T;
    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)


    plt.figure(1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x1, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x2, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    #-------------------------- EXERCISE 3 --------------------------#
    T_c2_c1 = T_c2_w @ T_w_c1;
    
    R_c2_c1 = T_c2_c1[:3, :3];
    Transl_c2_c1 = T_c2_c1[:3, 3];
   
    # Exercise 3.1: Homography definition
    Pi_c1 = np.loadtxt('Pi_1.txt')
    
    n_Pi_c1 = Pi_c1[:3];
    d = Pi_c1[3] / np.sqrt(Pi_c1[0] * Pi_c1[0] + Pi_c1[1] * Pi_c1[1] + Pi_c1[2] * Pi_c1[2]);
    
    H_c2_c1 = K_c @ (R_c2_c1 - (Transl_c2_c1.reshape(3,1) @ n_Pi_c1.reshape(1,3)) / Pi_c1[3]) @ np.linalg.inv(K_c)
    H_c2_c1 = H_c2_c1 / H_c2_c1[2][2]
    H_c2_c1_toPlot = H_c2_c1
    
    # Exercise 3.2: Point transfer visualization 
    drawHomography(20);
            
    # Exercise 3.3: 
    x1, y1, _ = np.loadtxt('x1FloorData.txt')
    x2, y2, _ = np.loadtxt('x2FloorData.txt')
    
    A = []
    for i in range(len(x1)):
        A.append([x1[i], y1[i], 1, 0, 0, 0, -x2[i]*x1[i], -x2[i]*y1[i], -x2[i]])
        A.append([0, 0, 0, x1[i], y1[i], 1, -y2[i]*x1[i], -y2[i]*y1[i], -y2[i]])
    A = np.array(A)
    print(A.shape)
    
    u, s, vh = np.linalg.svd(A);
    H_c2_c1_estimated = vh[-1].reshape(3, 3)
    
    H_c2_c1_estimated = H_c2_c1_estimated / H_c2_c1_estimated[2][2]
    H_c2_c1_toPlot = H_c2_c1_estimated
    
    drawHomography(21);
    
    