import tifffile
import numpy as np
import pickle
import cv2

def distort_model(params, x, y):
    fx, fy, cx, cy, k1, k2, k3, p1, p2 = params
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    objpoints = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], np.ones_like(y[:, np.newaxis])), axis=1)
    objpoints_rotated = np.matmul(objpoints, matrix)
    objpoints_projected = objpoints_rotated[:, :2] / (objpoints_rotated[:, 2:] + 1e-17)
    shift = objpoints_projected - np.array([cx, cy])

    x_shifted = shift[:, 0]
    y_shifted = shift[:, 1]
    r2 = x_shifted**2 + y_shifted**2
    x_distorted = x_shifted * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*x_shifted*y_shifted + p2*(r2 + 2*x_shifted**2) + cx
    y_distorted = y_shifted * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + p1*(r2 + 2*y_shifted**2) + 2*p2*x_shifted*y_shifted + cy
    return x_distorted, y_distorted

def undistort_coor(params):
    H, W = (10748, 14304)
    gty, gtx = np.mgrid[:H, :W]
    gtxy = np.c_[gtx.ravel(), gty.ravel()]
    x_undistorted, y_undistorted = distort_model(params['inv_undistort'], (gtxy[:,0]-W//2)/100, (gtxy[:,1]-H//2)/100)
    x_undistorted = x_undistorted*100 + W//2
    y_undistorted = y_undistorted*100 + H//2
    return x_undistorted, y_undistorted

def merge(wigner, group_mode, nshift=3):
    '''
    input:
        wigner: ( 180, ny_2, nx_2), dtype=torch.tensor
    output:
        merge_wigner: (172(20), ny_2*nshift, nx_2*nshift)
    '''
    order = [6, 5, 4, 7, 8, 3, 0, 1, 2]
    n, h, w = wigner.shape
    if group_mode == 1:  
        merged_wigner = np.zeros([n - nshift**2 + 1, h*nshift, w*nshift])
        for i in range(merged_wigner.shape[0]):
            wigner_tmp = wigner[i:i + nshift**2]
            wigner_tmp = np.roll(wigner_tmp, i%nshift**2, 0)
            merged_wigner[i] = wigner_tmp[order].reshape(nshift, nshift, h, w).transpose(2, 0, 3, 1).reshape(h*nshift, w*nshift)
    else:  
        merged_wigner = np.zeros([n//nshift**2, h*nshift, w*nshift])
        for i in range(merged_wigner.shape[0]):
            wigner_tmp = wigner[i * nshift**2:(i + 1) * nshift**2]
            merged_wigner[i] = wigner_tmp[order].reshape(nshift, nshift, h, w).transpose(2, 0, 3, 1).reshape(h*nshift, w*nshift)

    return merged_wigner

def register_ecc(img1, img2):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100,  1e-6)
    cc, warp_matrix = cv2.findTransformECC(img2, img1, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
    xshift = warp_matrix[0, 2]
    yshift = warp_matrix[1, 2]
    return yshift, xshift

if __name__ == '__main__':
    H, W, centerX, centerY = 10748, 14304, 7151, 5373
    crop_H, crop_W = 1995, 1995
    kk=1#kk is the parameter KK is a pre-corrected parameter associated with system parameters.
    grid_x = np.arange(centerX-crop_W//2+7, centerX+crop_W//2+1, 15, dtype=np.int16)
    grid_y = np.arange(centerY-crop_H//2+7, centerY+crop_H//2+1, 15, dtype=np.int16)
    gtx, gty = np.meshgrid(grid_x, grid_y)
    gtxy = np.c_[gtx.ravel(), gty.ravel()]
    
    with open("./undistort_params_dict_points_240620.pkl", 'rb') as file:
        params = pickle.load(file)
    
    x_undistorted, y_undistorted = distort_model(params['inv_undistort'], (gtxy[:,0]-W//2)/100, (gtxy[:,1]-H//2)/100)
    x_undistorted = np.round(x_undistorted*100 + W//2).astype(np.int16)
    y_undistorted = np.round(y_undistorted*100 + H//2).astype(np.int16)
    start_x = centerX-2415//2
    start_y = centerY-2415//2
    x_undistorted -= start_x
    y_undistorted -= start_y

    with open("output_fft.txt", "w") as f:
        for i in range(0, 25):
            raw_path = f"Y:/C2/B18_{i}.tiff"

            lf = tifffile.imread(raw_path)
            
            wdf0 = lf[:, y_undistorted, x_undistorted-2].reshape(lf.shape[0], 133, 133)
            wdf1 = lf[:, y_undistorted, x_undistorted+2].reshape(lf.shape[0], 133, 133)
            merged_wdf_0 = merge(wdf0, 0).astype(np.float32)
            merged_wdf_1 = merge(wdf1, 0).astype(np.float32)


            for frame in range(len(merged_wdf_0)):
                yshift, xshift = register_ecc(merged_wdf_0[frame], merged_wdf_1[frame])
                fractor_X = kk
                defocus_x = xshift / fractor_X
                print('frame:', frame+i*20, 'yshift:', yshift, ', xshift:', xshift, ', defocus_x:', defocus_x)
                f.write(f"{defocus_x}\n")
