#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Optical Flow
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Jesus Bermudez, Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv
import interpolation as ip
import matplotlib.pyplot as plt


def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross correlation values for a patch in a searching area.
    """
    # Complete the function
    i0 = patch
    sum0 = np.sum(i0)
    mean0 = sum0/i0.shape[0]/i0.shape[1]
    i0_subs = i0-mean0
    norm0 = np.linalg.norm(i0_subs)
    i0_n = i0_subs / norm0

    result = np.zeros(search_area.shape, dtype=float)
    margin_y = int(patch.shape[0]/2)
    margin_x = int(patch.shape[1]/2)

    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]
            # Implement the correlation
            sum = np.sum(i1)
            mean = sum/i1.shape[0]/i1.shape[1]
            i1_subs = i1-mean
            norm = np.linalg.norm(i1_subs)
            i1_n = i1_subs / norm
            mult = i0_n * i1_n
            result[i, j] = np.sum(mult)
    return result


def seed_estimation_NCC_single_point(img1_gray, img2_gray, i_img, j_img, patch_half_size: int = 5, searching_area_size: int = 100):

    # Attention!! we are not checking the padding
    patch = img1_gray[i_img - patch_half_size:i_img + patch_half_size + 1, j_img - patch_half_size:j_img + patch_half_size + 1]

    i_ini_sa = i_img - int(searching_area_size / 2)
    i_end_sa = i_img + int(searching_area_size / 2) + 1
    j_ini_sa = j_img - int(searching_area_size / 2)
    j_end_sa = j_img + int(searching_area_size / 2) + 1

    search_area = img2_gray[i_ini_sa:i_end_sa, j_ini_sa:j_end_sa]
    result = normalized_cross_correlation(patch, search_area)

    iMax, jMax = np.where(result == np.amax(result))

    i_flow = i_ini_sa + iMax[0] - i_img
    j_flow = j_ini_sa + jMax[0] - j_img

    return i_flow, j_flow

def lucasKanade(img0_gray, img1_gray, ini_flow: np.array, points_selected: np.array, patch_half_size: int = 5):
    patch_size=(patch_half_size*2+1)*(patch_half_size*2+1)
    #flow_shape = ini_flow.shape
    #print(flow_shape)
    new_flow = np.zeros(ini_flow.shape, dtype=float)
    
    threshold = 0.000001
    for i in range(0, points_selected.shape[0]):
        #print("i: ",i)
        
        orig_pixel = points_selected[i,:]
        xmin = orig_pixel[0] - patch_half_size
        ymin = orig_pixel[1] - patch_half_size
        patch_coord_img0 = np.zeros((patch_half_size*2+1,patch_half_size*2+1,2),dtype=int)
        patch_coord_img0_vector = np.zeros((patch_half_size*2+1,patch_half_size*2+1,2),dtype=float)
        patch_coord_img0_vector = patch_coord_img0_vector.reshape(patch_size, 2)
        for j in range(0, patch_half_size*2+1):
            for k in range(0, patch_half_size*2+1):
                patch_coord_img0[j, k, 0] = xmin + j
                patch_coord_img0[j, k, 1] = ymin + k
        patch_coord_img0 = patch_coord_img0.reshape(patch_size, 2)
        patch_coord_img0_vector[:,0] = patch_coord_img0[:,1]
        patch_coord_img0_vector[:,1] = patch_coord_img0[:,0]
        patch_coord_img0_vector = np.array(patch_coord_img0_vector, dtype=float)
        #print("patch_coord_img0: ",patch_coord_img0)
        patch_0 = ip.int_bilineal(img0_gray, patch_coord_img0_vector)
        
        delta_u = np.ones(ini_flow.shape[0])
        u = ini_flow[i]
        
        grads = ip.numerical_gradient(img0_gray, patch_coord_img0_vector)
        
        A = np.zeros((2,2), dtype= float)
        b = np.zeros((2,1), dtype=float)
        Ix = grads[:, 0]
        Iy = grads[:, 1] 
        A[0,0]= np.sum(Ix*Ix)
        A[0,1]= np.sum(Ix*Iy)
        A[1,0]= np.sum(Ix*Iy)
        A[1,1]= np.sum(Iy*Iy)
        
        
        detA = np.linalg.det(A)

        if (detA!=0):
            #print(delta_u)
            #iteration = 0
            patch_coord_img1_vector = np.zeros((patch_half_size*2+1,patch_half_size*2+1,2),dtype=float)
            patch_coord_img1_vector = patch_coord_img1_vector.reshape(patch_size, 2)
            while (np.sqrt(np.sum(delta_u ** 2)) >= threshold):
                #iteration+=1
                #print("Iteration: ",iteration)
                patch_coord_img1 = patch_coord_img0 + u
                
                patch_coord_img1_vector[:,0] = patch_coord_img1[:,1]
                patch_coord_img1_vector[:,1] = patch_coord_img1[:,0]
                patch_coord_img1_vector = np.array(patch_coord_img1_vector, dtype=float)
                #print(patch_coord_img1_vector[:, 0])
                #print(patch_coord_img0_vector[:, 0])
                patch_1 = ip.int_bilineal(img1_gray, patch_coord_img1_vector)
                e = patch_1 - patch_0
                b[0] = -np.sum(e * Ix)
                b[1] = -np.sum(e * Iy) 
                 
                delta_u = np.linalg.solve(A, b)
                delta_u = delta_u.reshape((2,))
                #print("A: ",A)
                #print("b: ",b)
                #print("delta_u: ",delta_u)
                
                u = u + delta_u
                
                
                #print("new u: ", u)
            #print("def u: ", u)
            new_flow[i] = u  
            #print(new_flow)         
        else:
            print("Matrix A cannot be inverted")
    
    return new_flow

def read_flo_file(filename, verbose=False):
        """
        Read from .flo optical flow file (Middlebury format)
        :param flow_file: name of the flow file
        :return: optical flow data in matrix

        adapted from https://github.com/liruoteng/OpticalFlowToolkit/

        """
        f = open(filename, 'rb')
        magic = np.fromfile(f, np.float32, count=1)
        data2d = None

        if 202021.25 != magic:
            raise TypeError('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            if verbose:
                print("Reading %d x %d flow file in .flo format" % (h, w))
            data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
            # reshape data into 3D array (columns, rows, channels)
            data2d = np.resize(data2d, (h[0], w[0], 2))
        f.close()
        return data2d




if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    unknownFlowThresh = 1e9
    flow_12 = read_flo_file("flow10.flo", verbose=True)
    binUnknownFlow = flow_12 > unknownFlowThresh

    img1 = read_image("frame10.png")
    img2 = read_image("frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points
    points_selected = np.loadtxt('points_selected.txt')
    points_selected = points_selected.astype(int)

    template_size_half = 5
    searching_area_size: int = 15

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(seed_optical_flow_sparse)
    new_flow = lucasKanade(img1_gray, img2_gray, seed_optical_flow_sparse, points_selected, template_size_half)
    print("new_flow: ", new_flow)

    ## Sparse optical flow
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_est_sparse = seed_optical_flow_sparse#[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_sparse = flow_est_sparse - flow_gt
    error_sparse_norm = np.sqrt(np.sum(error_sparse ** 2, axis=1))


    # Plot results for sparse optical flow
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow')
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error of first estimation respect to GT')

    #---------------------------------------------------------------------------------------------------------------
    
    ## Sparse optical flow
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_est_sparse = new_flow#[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)]
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_sparse = flow_est_sparse - flow_gt
    error_sparse_norm = np.sqrt(np.sum(error_sparse ** 2, axis=1))


    # Plot results for sparse optical flow
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow')
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error of new estimation respect to GT')
    plt.show()

