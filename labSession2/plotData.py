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
    
def Capture_Event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_0 = np.array([x, y]);
        
        
    return x_0;
  
def on_click(event):
    if event.button == 1:  # Left mouse button
        plt.figure(4)
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.title('Image 2 - Epipolar Lines')
        plt.draw()  # We update the figure display
        print(f'You clicked at ({event.xdata}, {event.ydata})')
        x_0 = np.array([event.xdata, event.ydata, 1])
        l_xi_1 = np.dot(F_21, x_0);
        drawLine(l_xi_1, 'g-', 1)
        
                

def drawEipolarLine (): # Draw epipolar line of a clicked point
    fig = plt.figure(3)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.title('Image 1 - Click a point')
    plt.draw()  # We update the figure display
    fig.canvas.mpl_connect('button_press_event', on_click)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return;



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
    
    #-------------------------- EXERCISE 2 --------------------------#
    
    # Exercise 2.1: epipolar lines with given F_c2_c1
    drawEipolarLine();
    
    # Exercise 2.2: Calculate E_c2_c1 and F_c2_c1
    
    T_c2_c1 = T_c2_w @ T_w_c1;
    
    R_c2_c1 = T_c2_c1[:3, :3];
    Transl_c2_c1 = T_c2_c1[:3, 3];
    
    Transl_c2_c1_mod = np.array([[0, -Transl_c2_c1[2], Transl_c2_c1[1]],
                                 [Transl_c2_c1[2], 0, -Transl_c2_c1[0]],
                                 [-Transl_c2_c1[1], Transl_c2_c1[0], 0]]);
    
    E_c2_c1 = np.dot(Transl_c2_c1_mod, R_c2_c1);
    
    F_c2_c1 = np.linalg.inv(K_c).T @ E_c2_c1 @ np.linalg.inv(K_c);
    
    F_21 = F_c2_c1;
    
    drawEipolarLine();
    
    # Exercise 2.3: Compute F by estimation with 8 correspondences
    x1Data = np.loadtxt('x1Data.txt')
    x2Data = np.loadtxt('x2Data.txt')
    A=[]
    k=0
    print(x1Data.shape)
    for i in range (x1Data.shape[1]):
        x1,y1=x1Data[:,i]
        x2,y2=x2Data[:,i]
        A.append([x1*x2,y1*x2,x2,x1*y2,y1*y2,y2,x1,y1,1])
    A=np.array(A)
    print(A.shape)
    
    _,_,V=np.linalg.svd(A)
    F_estimated=V[-1].reshape(3,3)
    print(F_estimated)
    pt1=(x1Data[0,0],x1Data[1,0],1)
    pt2=(x1Data[0,1],x1Data[1,1],1)
    pt3=(x1Data[0,2],x1Data[1,2],1)
    pt4=(x1Data[0,3],x1Data[1,3],1)
    pt5=(x1Data[0,4],x1Data[1,4],1)
    l_1_1=F_estimated@pt1
    l_1_2=F_estimated@pt2
    l_1_3=F_estimated@pt3
    l_1_4=F_estimated@pt4
    l_1_5=F_estimated@pt5
    e_1=np.dot(F_estimated.T,l_1_1)
    e_1=e_1/e_1[2]
    print(e_1)
    plt.figure(5)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(e_1[0], e_1[1],'rx', markersize=10)
    plt.title('Image 2 - Epipolar Lines Estimated')
    plt.draw()  # We update the figure display
    drawLine(l_1_1, 'g-', 1)
    drawLine(l_1_2, 'b-', 1)
    drawLine(l_1_3, 'r-', 1)
    drawLine(l_1_4, 'y-', 1)
    drawLine(l_1_5, 'p-', 1)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    


    
   