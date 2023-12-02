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

# Forward Kannala model
def projectionModel(X, K_c, D):
    phi = np.arctan2(X[1], X[0]);
    R = np.sqrt(X[0] ** 2 + X[1] ** 2);
    theta = np.arctan2(R, X[2]);
    d = theta + D[0] * theta**3 + D[1] * theta**5 + D[2] * theta**7 + D[3] * theta**9;
    x = np.array([d * np.cos(phi), d * np.sin(phi), 1]);
    u = K_c @  x.T;
    
    return u;
    
# Backward Kannala model return x and v
def unprojectionModel(u, K_c, D):
    x = np.linalg.inv(K_c) @ u.T;
    phi = np.arctan2(x[1], x[0]);
    d = np.sqrt((x[0]**2 + x[1]**2) / x[2]**2);
    roots = np.roots([D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -d]);
    theta = np.real(roots[np.isreal(roots)])[0];
    v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]);
    
    return v;

def triangulatePoint(v_1, v_2, T_c1w, T_c2w):
    # pi_sym_cam_ref
    pi_sym_1_1 = np.array([-v_1[1], v_1[0], 0, 0]).T;
    pi_per_1_1 = np.array([-v_1[2] * v_1[0], -v_1[2] * v_1[1], v_1[0]**2 + v_1[1]**2, 0]).T;
    pi_sym_2_2 = np.array([-v_2[1], v_2[0], 0, 0]).T;
    pi_per_2_2 = np.array([-v_2[2] * v_2[0], -v_2[2] * v_2[1], v_2[0]**2 + v_2[1]**2, 0]).T;
    
    pi_sym_1_w = T_c1w.T @ pi_sym_1_1;
    pi_per_1_w = T_c1w.T @ pi_per_1_1;
    pi_sym_2_w = T_c2w.T @ pi_sym_2_2;
    pi_per_2_w = T_c2w.T @ pi_per_2_2;
    
    A = np.array([pi_sym_1_w.T, pi_per_1_w.T, pi_sym_2_w.T, pi_per_2_w.T])
    
    _, _, vt = np.linalg.svd(A);
    X_w = vt[-1, :];
    X_w = X_w / X_w[3];
    
    # X_w is in world ref, be careful so the X_1 is given in cam1 ref
    
    return X_w;
    

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # Load data
    D1_k_array = np.loadtxt('D1_k_array.txt')
    D2_k_array = np.loadtxt('D2_k_array.txt')
    K_1 = np.loadtxt('K_1.txt')
    K_2 = np.loadtxt('K_2.txt')
    T_leftRight = np.loadtxt('T_leftRight.txt')
    T_wAwB_gt = np.loadtxt('T_wAwB_gt.txt')
    T_wAwB_seed = np.loadtxt('T_wAwB_seed.txt')
    T_wc1 = np.loadtxt('T_wc1.txt')
    T_wc2 = np.loadtxt('T_wc2.txt') 
    x1Data = np.loadtxt('x1.txt')
    x2Data = np.loadtxt('x2.txt')
    x3Data = np.loadtxt('x3.txt')
    x4Data = np.loadtxt('x4.txt')
    
    T_c1w = np.linalg.inv(T_wc1);
    T_c2w = np.linalg.inv(T_wc2);
    T_c2_c1 = T_c2w @ T_wc1;
    
    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread('fisheye1_frameA.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('fisheye1_frameB.png'), cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(cv2.imread('fisheye2_frameA.png'), cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(cv2.imread('fisheye2_frameB.png'), cv2.COLOR_BGR2RGB)
    
    ############################ EXERCISE 2 ############################
    #-------------------------- Ex 2.1: projection --------------------------#
    X_1_c1 = np.array([3, 2, 10, 1]).T;
    X_2_c1 = np.array([-5, 6, 7, 1]).T;
    X_3_c1 = np.array([1, 5, 14, 1]).T;
    
    u_1_c1 = projectionModel(X_1_c1, K_1, D1_k_array);
    u_2_c1 = projectionModel(X_2_c1, K_1, D1_k_array);
    u_3_c1 = projectionModel(X_3_c1, K_1, D1_k_array);
    
    #move point to cam2
    X_1_c2 = T_c2_c1 @ X_1_c1;
    X_2_c2 = T_c2_c1 @ X_2_c1;
    X_3_c2 = T_c2_c1 @ X_3_c1;
    
    u_1_c2 = projectionModel(X_1_c2, K_2, D2_k_array);
    u_2_c2 = projectionModel(X_2_c2, K_2, D2_k_array);
    u_3_c2 = projectionModel(X_3_c2, K_2, D2_k_array);
    
    #-------------------------- Ex 2.1: unprojection --------------------------#
    v_1_c1 = unprojectionModel(u_1_c1, K_1, D1_k_array);
    v_2_c1 = unprojectionModel(u_2_c1, K_1, D1_k_array);
    v_3_c1 = unprojectionModel(u_3_c1, K_1, D1_k_array);
    
    v_1_c2 = unprojectionModel(u_1_c2, K_2, D2_k_array);
    v_2_c2 = unprojectionModel(u_2_c2, K_2, D2_k_array);
    v_3_c2 = unprojectionModel(u_3_c2, K_2, D2_k_array);
    
    #-------------------------- Ex 2.2: triangulation --------------------------#
    # Toy example
    X_w_true = T_wc1 @ X_1_c1 
    X_w_true = X_w_true / X_w_true[3];
    X_w = triangulatePoint(v_1_c1, v_1_c2, T_c1w, T_c2w);
    
    # Full set of 3d points x1Data and x2Data
    X_poseA_w = [];
    for i in range(x1Data.shape[1]):
        u_c1 = x1Data[:, i];
        u_c2 = x2Data[:, i];
        
        v_c1 = unprojectionModel(u_c1, K_1, D1_k_array);
        v_c2 = unprojectionModel(u_c2, K_2, D2_k_array);
        
        x_poseA_w = triangulatePoint(v_c1, v_c2, T_c1w, T_c2w);
        X_poseA_w.append(x_poseA_w);
    X_poseA_w = np.array(X_poseA_w).T;
    
    # Draw 2d matches
    plt.figure(0)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plt.title('Image 1')
    plt.draw()

    plt.figure(1)
    plt.imshow(img3, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plt.title('Image 2')
    plt.draw()
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()
    
    # Draw 3D
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2')
    
    ax.scatter(X_poseA_w[0, :], X_poseA_w[1, :], X_poseA_w[2, :], marker='.', c="green")
    
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()
        
    
    # Full set of 3d points x3Data and x4Data
    X_poseB_w = [];
    for i in range(x3Data.shape[1]):
        u_c1 = x3Data[:, i];
        u_c2 = x4Data[:, i];
        
        v_c1 = unprojectionModel(u_c1, K_1, D1_k_array);
        v_c2 = unprojectionModel(u_c2, K_2, D2_k_array);
        
        x_poseB_w = triangulatePoint(v_c1, v_c2, T_c1w, T_c2w);
        X_poseB_w.append(x_poseB_w);
    X_poseB_w = np.array(X_poseB_w).T;
    
     # Draw 2d matches
    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plt.title('Image 3')
    plt.draw()

    plt.figure(3)
    plt.imshow(img4, cmap='gray', vmin=0, vmax=255)
    plt.plot(x4Data[0, :], x4Data[1, :], 'rx')
    plt.title('Image 4')
    plt.draw()
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()
    
    # Draw 3D
    # Draw 3D
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2')
    
    ax.scatter(X_poseB_w[0, :], X_poseB_w[1, :], X_poseB_w[2, :], marker='.', c="green")
    
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()
    
    
    #-------------------------- CLASS EXAMPLE --------------------------#
    X_test_c1 = np.array([4, 0 ,10, 1]);
    X_test_c2 = np.array([-8, 0 ,10, 1]);
    v_test_c1 = np.array([4, 0, 10]);
    v_test_c2 = np.array([-8, 0, 10]);
    
    T_test_c1c2 = np.array([[1, 0, 0, 12],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]);
    T_test_wc1 = np.array([[1, 0, 0, -6],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]);
    T_test_wc2 = np.array([[1, 0, 0, 6],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]);
    
    T_test_c1w = np.linalg.inv(T_test_wc1);
    T_test_c2w = np.linalg.inv(T_test_wc2);
    
    X_w = triangulatePoint(v_test_c1, v_test_c2, T_test_c1w, T_test_c2w)
    X_w_test = T_test_wc1 @ X_test_c1;
    X_w_test = T_test_wc2 @ X_test_c2;
    
    a = 0;