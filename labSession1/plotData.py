#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 1
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 14 September 2022
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
from line2DFittingSVD import drawLine


# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
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

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    R_w_c1 = np.loadtxt('R_w_c1.txt')
    R_w_c2 = np.loadtxt('R_w_c2.txt')

    t_w_c1 = np.loadtxt('t_w_c1.txt')
    t_w_c2 = np.loadtxt('t_w_c2.txt')

    # Translation matrix
    T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
    T_w_c2 = ensamble_T(R_w_c2, t_w_c2)
    
    T_c1_w = np.linalg.inv(T_w_c1);
    T_c2_w = np.linalg.inv(T_w_c2);

    # Camera calibration
    K_c = np.loadtxt('K.txt')
    
    # Canonical perspective projection matrix
    P_canonical = np.array([[1, 0, 0 ,0], [0, 1, 0 ,0], [0, 0, 1, 0]]);
    
    # Projection matrixes
    P_c1 = K_c @ P_canonical @ T_c1_w;
    P_c2 = K_c @ P_canonical @ T_c2_w;

    # Points to calculate
    X_A = np.array([3.44, 0.80, 0.82])
    X_B = np.array([4.20, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])
    X_D = np.array([3.55, 0.60, 0.82])
    X_E = np.array([-0.01, 2.6, 1.21])

    print(np.array([[3.44, 0.80, 0.82]]).T) #transpose need to have dimension 2
    print(np.array([3.44, 0.80, 0.82]).T) #transpose does not work with 1 dim arrays

    # Example of transpose (need to have dimension 2)  and concatenation in numpy
    X_w = np.vstack((np.hstack((np.reshape(X_A,(3,1)), np.reshape(X_B,(3,1)), np.reshape(X_C,(3,1)), 
                                np.reshape(X_D,(3,1)), np.reshape(X_E,(3,1)))), np.ones((1, 5))))

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
    plotLabelled3DPoints(ax, X_w, ['A', 'B', 'C', 'D', 'E'], 'r', (-0.3, -0.3, 0.1)) # For plotting with labels (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    #Drawing a 3D segment
    draw3DLine(ax, X_A, X_C, '--', 'k', 1)

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread("Image1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("Image2.jpg"), cv2.COLOR_BGR2RGB)
    
    print(X_w[:, 0])
    
    x_A_2d_C1 = P_c1 @ X_w[:, 0];
    x_A_2d_C1_n = x_A_2d_C1 / x_A_2d_C1[2]; 
    x_B_2d_C1 = P_c1 @ X_w[:, 1];
    x_B_2d_C1_n = x_B_2d_C1 / x_B_2d_C1[2]; 
    x_C_2d_C1 = P_c1 @ X_w[:, 2];
    x_C_2d_C1_n = x_C_2d_C1 / x_C_2d_C1[2]; 
    x_D_2d_C1 = P_c1 @ X_w[:, 3];
    x_D_2d_C1_n = x_D_2d_C1 / x_D_2d_C1[2]; 
    x_E_2d_C1 = P_c1 @ X_w[:, 4];
    x_E_2d_C1_n = x_E_2d_C1 / x_E_2d_C1[2]; 
    
    x_A_2d_C2 = P_c2 @ X_w[:, 0];
    x_A_2d_C2_n = x_A_2d_C2 / x_A_2d_C2[2]; 
    x_B_2d_C2 = P_c2 @ X_w[:, 1];
    x_B_2d_C2_n = x_B_2d_C2 / x_B_2d_C2[2]; 
    x_C_2d_C2 = P_c2 @ X_w[:, 2];
    x_C_2d_C2_n = x_C_2d_C2 / x_C_2d_C2[2]; 
    x_D_2d_C2 = P_c2 @ X_w[:, 3];
    x_D_2d_C2_n = x_D_2d_C2 / x_D_2d_C2[2]; 
    x_E_2d_C2 = P_c2 @ X_w[:, 4];
    x_E_2d_C2_n = x_E_2d_C2 / x_E_2d_C2[2]; 

    x1 = np.array([[x_A_2d_C1_n[0], x_B_2d_C1_n[0], x_C_2d_C1_n[0], x_D_2d_C1_n[0], x_E_2d_C1_n[0]],
                   [x_A_2d_C1_n[1], x_B_2d_C1_n[1], x_C_2d_C1_n[1], x_D_2d_C1_n[1], x_E_2d_C1_n[1]]]);
    
    x2 = np.array([[x_A_2d_C2_n[0], x_B_2d_C2_n[0], x_C_2d_C2_n[0], x_D_2d_C2_n[0], x_E_2d_C2_n[0]],
                   [x_A_2d_C2_n[1], x_B_2d_C2_n[1], x_C_2d_C2_n[1], x_D_2d_C2_n[1], x_E_2d_C2_n[1]]]);
    
    l_ab_1 = np.cross(x_A_2d_C1_n, x_B_2d_C1_n);
    l_cd_1 = np.cross(x_C_2d_C1_n, x_D_2d_C1_n);
    
    l_ab_2 = np.cross(x_A_2d_C2_n, x_B_2d_C2_n);
    l_cd_2 = np.cross(x_C_2d_C2_n, x_D_2d_C2_n);
    
    p_12_1 = np.cross(l_ab_1, l_cd_1);
    p_12_1_n = p_12_1 / p_12_1[2];
    p_12_2 = np.cross(l_ab_2, l_cd_2);
    p_12_2_n = p_12_2 / p_12_2[2];
    
    # Vector ab in 3D
    
    V_AB = X_B - X_A;
    p_AB_inf = np.array([V_AB[0], V_AB[1], V_AB[2], 0]).T;
    
    p_AB_inf_2d_1 = P_c1 @ p_AB_inf;
    p_AB_inf_2d_1_n = p_AB_inf_2d_1 / p_AB_inf_2d_1[2];
    p_AB_inf_2d_2 = P_c2 @ p_AB_inf;
    p_AB_inf_2d_2_n = p_AB_inf_2d_2/ p_AB_inf_2d_2[2];
    
    A = X_w.T[:4,:];
    
    u, s, vh = np.linalg.svd(A);
    plane = vh[-1, :];
    
    dist_plane_A = abs(plane[0] * X_A[0] + plane[1] * X_A[1] + plane[2] * X_A[2] + plane[3]) / np.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    dist_plane_B = abs(plane[0] * X_B[0] + plane[1] * X_B[1] + plane[2] * X_B[2] + plane[3]) / np.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    dist_plane_C = abs(plane[0] * X_C[0] + plane[1] * X_C[1] + plane[2] * X_C[2] + plane[3]) / np.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    dist_plane_D = abs(plane[0] * X_D[0] + plane[1] * X_D[1] + plane[2] * X_D[2] + plane[3]) / np.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    dist_plane_E = abs(plane[0] * X_E[0] + plane[1] * X_E[1] + plane[2] * X_E[2] + plane[3]) / np.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);


    plt.figure(1)
    plt.imshow(img1)
    plt.plot(x1[0, :], x1[1, :],'+r', markersize=15)
    plt.plot(p_12_1_n[0], p_12_1_n[1],'.b', markersize=10)
    plt.plot(p_AB_inf_2d_1_n[0], p_AB_inf_2d_1_n[1],'.g', markersize=5)
    plotLabeledImagePoints(x1, ['a', 'b', 'c', 'd', 'e'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    plotNumberedImagePoints(x1, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    drawLine(l_ab_1, 'g-', 1)
    drawLine(l_cd_1, 'g-', 1)
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    plt.figure(2)
    plt.imshow(img2)
    plt.plot(x2[0, :], x2[1, :],'+r', markersize=15)
    plt.plot(p_12_2_n[0], p_12_2_n[1],'.b', markersize=10)
    plt.plot(p_AB_inf_2d_2_n[0], p_AB_inf_2d_2_n[1],'.g', markersize=5)
    plotLabeledImagePoints(x2, ['a', 'b', 'c', 'd', 'e'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    plotNumberedImagePoints(x2, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 2')
    drawLine(l_ab_2, 'g-', 1)
    drawLine(l_cd_2, 'g-', 1)
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
        