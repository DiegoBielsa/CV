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
    
def Capture_Event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_0 = np.array([x, y]);
        
        
    return x_0;
  
def on_click_epipolar(event):
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
    fig.canvas.mpl_connect('button_press_event', on_click_epipolar)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return;

def on_click_homography(event):
    if event.button == 1:  # Left mouse button
        plt.figure(7)
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.title('Image 2 - Homography')
        plt.draw()  # We update the figure display
        print(f'You clicked at ({event.xdata}, {event.ydata})')
        p1 = np.array([event.xdata, event.ydata, 1])
        p2 = np.dot(H_c2_c1, p1) # apply homography
        p2 /= p2[2]
        plt.plot(p2[0], p2[1],'rx', markersize=10)

def drawHomography (): # Draw epipolar line of a clicked point
    fig = plt.figure(6)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.title('Image 1 - Click a point on the ground plane')
    plt.draw()  # We update the figure display
    fig.canvas.mpl_connect('button_press_event', on_click_homography)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    return;
    

def display_homography_correspondances(image1, image2, homography_matrix):
    def show_corresponding_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point1 = np.array([x, y, 1])
            point2 = np.dot(homography_matrix, point1)
            point2 /= point2[2]
            cv2.circle(image2, (int(point2[0]), int(point2[1])), 5, (0, 0, 255), -1)
            cv2.imshow('Image 2', image2)

    cv2.imshow('Image 1', image1)
    cv2.setMouseCallback('Image 1', show_corresponding_point)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    #drawEipolarLine();
    
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
    
    #drawEipolarLine();
    
    # Exercise 2.3: Compute F by estimation with 8 correspondences
    x1Data = np.loadtxt('x1Data.txt')
    x2Data = np.loadtxt('x2Data.txt')
    A=[]
    k=0
    print(x1Data.shape)
    for i in range (x1Data.shape[1]):
        x0, y0 = x1Data[:, i]
        x1, y1 = x2Data[:, i]
        A.append([x0*x1, y0*x1, x1, x0*y1, y0*y1, y1, x0, y0, 1])
    A = np.array(A)
    print(A.shape)
    
    _, _, V = np.linalg.svd(A)
    F_c2_c1_estimated = V[-1].reshape(3, 3)
    rank = np.linalg.matrix_rank(F_c2_c1_estimated)
    print(rank)
    print(F_c2_c1_estimated)
    U, S, V = np.linalg.svd(F_c2_c1_estimated)
    S[2:]=0
    F_c2_c1_estimated = np.dot(U,np.dot(np.diag(S),V))
    rank = np.linalg.matrix_rank(F_c2_c1_estimated)
    print(rank)
    print(F_c2_c1_estimated)
    pt1 = (x1Data[0,0], x1Data[1,0],1)
    pt2 = (x1Data[0,1], x1Data[1,1],1)
    pt3 = (x1Data[0,2], x1Data[1,2],1)
    pt4 = (x1Data[0,3], x1Data[1,3],1)
    pt5 = (x1Data[0,4], x1Data[1,4],1)
    
    F_21 = F_c2_c1_estimated;
    
    drawEipolarLine();
    
    l_1_1 = F_c2_c1_estimated @ pt1
    l_1_2 = F_c2_c1_estimated @ pt2
    l_1_3 = F_c2_c1_estimated @ pt3
    l_1_4 = F_c2_c1_estimated @ pt4
    l_1_5 = F_c2_c1_estimated @ pt5
    
    u, s, vh = np.linalg.svd(F_c2_c1_estimated.T);
    e_2 = vh[-1, :];
    e_2 = e_2 / e_2[2]
    print(e_2)
    plt.figure(5)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(e_2[0], e_2[1],'rx', markersize=10)
    plt.title('Image 2 - Epipolar Lines Estimated')
    plt.draw()  # We update the figure display
    drawLine(l_1_1, 'g-', 1)
    drawLine(l_1_2, 'b-', 1)
    drawLine(l_1_3, 'r-', 1)
    drawLine(l_1_4, 'y-', 1)
    drawLine(l_1_5, 'p-', 1)
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    
    # Exercise 2.4: Estimate camera poses from F21
    
    # Diego
    E_c2_c1_estimated = (K_c.T) @ F_c2_c1_estimated @ K_c
    U, S, Vt = np.linalg.svd(E_c2_c1_estimated)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # R +90 and +t
    R0 = U @ W @ Vt;
    t0 = U[:, 2]
    # R +90 and -t
    R1 = U @ W @ Vt;
    t1 = -U[:, 2]
    # R -90 and +t
    R2 = U @ W.T @ Vt;
    t2 = U[:, 2]
    # R -90 and -t
    R3 = U @ W.T @ Vt;
    t3 = -U[:, 2]
    
    # Triangulation of one point (The first one for example)
    
    x0, y0 = x1Data[:, 0]
    x1, y1 = x2Data[:, 0]
    A = np.array([[P_c1[2][0] * x0 - P_c1[0][0], P_c1[2][1] * x0 - P_c1[0][1], P_c1[2][2] * x0 - P_c1[0][2], P_c1[2][3] * x0 - P_c1[0][3]],
                 [P_c1[2][0] * y0 - P_c1[1][0], P_c1[2][1] * y0 - P_c1[1][1], P_c1[2][2] * y0 - P_c1[1][2], P_c1[2][3] * y0 - P_c1[1][3]],
                 [P_c1[2][0] * x1 - P_c1[0][0], P_c1[2][1] * x1 - P_c1[0][1], P_c1[2][2] * x1 - P_c1[0][2], P_c1[2][3] * x1 - P_c1[0][3]],
                 [P_c1[2][0] * y1 - P_c1[1][0], P_c1[2][1] * y1 - P_c1[1][1], P_c1[2][2] * y1 - P_c1[1][2], P_c1[2][3] * y1 - P_c1[1][3]]]);
    
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :];
    X /= X[3];
    
    # Now lets calculate if the point is seen with both cameras
    T_c2_c1_estimated = np.eye(4)
    T_c2_c1_estimated[:3, :3] = R0
    T_c2_c1_estimated[:3, 3] = t0
    P_cam1 = K_c @ (T_c2_c1_estimated @ X.T)
    P_cam2 = K_c @ X
    
    P_cam1_test = T_c1_w @ (R0 @ X)
    P_cam2__test = T_c2_w @ X

    # Alejandro
    """
    E_c2_c1_estimated=(K_c.T)@F_c2_c1_estimated@K_c
    U,S,V=np.linalg.svd(E_c2_c1_estimated)
    print(S)
    rank = np.linalg.matrix_rank(E_c2_c1_estimated)  
    print(E_c2_c1_estimated)
    print(rank)
    U,S,V=np.linalg.svd(E_c2_c1_estimated)
    S=np.array([[1,0,0],[0,1,0],[0,0,0]])
    E_c2_c1_estimated = np.dot(U,np.dot(S,V))
    rank = np.linalg.matrix_rank(E_c2_c1_estimated)   
    print(E_c2_c1_estimated)
    print(rank)
    U,S,V=np.linalg.svd(E_c2_c1_estimated)
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1=U@W@V
    R2=U@(W.T)@V
    T1=U@S@U.T
    T2=-U@S@U.T
    
    
    print(R1.shape)
    
    Pose_1=np.hstack((R1,(T1.T).reshape(-1,1)))
    Pose_2=np.hstack((R2,(T1.T).reshape(-1,1)))
    Pose_3=np.hstack((R1,(T2.T).reshape(-1,1)))
    Pose_4=np.hstack((R2,(T2.T).reshape(-1,1)))
    print(Pose_1)
    I_matrix=np.eye(3)
    O_matrix=np.zeros((1,3))
    I_matrix_2=np.hstack(I_matrix,O_matrix)
    P1=K_c@I_matrix_2 
    P2_1=K_c@Pose_1
    P2_2=K_c@Pose_2 
    P2_3=K_c@Pose_3
    P2_4=K_c@Pose_4
    Points_2=[]

    r,c=x2Data.shape
    for i in range (r):
        new_row=[]
        for j in range(c+1):
            new_row.append()

    X_1=np.linalg.inv(P2_1)@Points_2
    X_2=np.linalg.inv(P2_2)@Points_2
    X_3=np.linalg.inv(P2_3)@Points_2
    X_4=np.linalg.inv(P2_4)@Points_2
    """
    # Exercise 2.5: Visualization and comparison
    T_c1_c2=T_c1_w@T_w_c2

    fig3D = plt.figure(6)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1_estimated')
    drawRefSystem(ax, T_c1_c2_estimated, '-', 'C2_estimated')
    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_c1_c2, '-', 'C2')


    ax.scatter(X_3D_estimated[0, :], X_3D_estimated[1, :], X_3D_estimated[2, :], marker='.')
    plotNumbered3DPoints(ax, X_3D_estimated, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    P_C2_C1=T_c1_c2@[0,0,0,1]
    P_C2_C1_estimated=T_c1_c2_estimated@[0,0,0,1]
    
    #-------------------------- EXERCISE 3 --------------------------#
   
    # Exercise 3.1: Homography definition
    Pi_c1 = np.loadtxt('Pi_1.txt')
    
    n_Pi_c1 = Pi_c1[:3];
    d = Pi_c1[3] / np.sqrt(Pi_c1[0] * Pi_c1[0] + Pi_c1[1] * Pi_c1[1] + Pi_c1[2] * Pi_c1[2]);
    
    H_c2_c1 = K_c @ (R_c2_c1 - (Transl_c2_c1.reshape(3,1) @ n_Pi_c1.reshape(1,3)) / Pi_c1[3]) @ np.linalg.inv(K_c)
    H_c2_c1 = H_c2_c1 / H_c2_c1[2][2]
    H_c2_c1_toPlot = H_c2_c1
    
    # Exercise 3.2: Point transfer visualization DO I HAVE TO MAKE THEM RANFOMLY?¿
    #drawHomography();
            
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
    
    drawHomography();
    
    