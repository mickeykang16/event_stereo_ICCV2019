import sys
import os
import numpy as np
import cv2
import yaml
import torch
import flow_vis

def png_name(i: int): 
    file_name = '%06d.png' % i
    return file_name
def read_mvsec_projection(intrinsic_fn):
    data = {}
    resolution = None
    with open(intrinsic_fn, 'r') as fp:
        yml = yaml.load(fp, Loader=yaml.FullLoader)
        # for line in fp.readlines():
        #     if line.find('P_rect_02') > -1:
                # line = line[11:]
                # values = line.rstrip().split(' ')
        values = yml['cam0']['projection_matrix']
        camera = '02'
        key = 'K_cam{}'.format(camera)
        inv_key = 'inv_K_cam{}'.format(int(camera))
        # matrix = np.array([float(values[i]) for i in range(len(values))])
        # K = np.eye(3)
        # K[0, 0] = matrix[0]
        # K[1, 1] = matrix[1]
        # K[0, 2] = matrix[2]
        # K[1, 2] = matrix[3]
        K = np.array(values)[:, :3]
        item = {
            key: K,
            inv_key: np.linalg.pinv(K)
        }
        data['{}'.format(camera)] = item

    return K
def read_mvsec_intrinsic(intrinsic_fn):
    data = {}
    resolution = None
    with open(intrinsic_fn, 'r') as fp:
        yml = yaml.load(fp, Loader=yaml.FullLoader)
        # for line in fp.readlines():
        #     if line.find('P_rect_02') > -1:
                # line = line[11:]
                # values = line.rstrip().split(' ')
        values = yml['cam0']['intrinsics']
        camera = '02'
        key = 'K_cam{}'.format(camera)
        inv_key = 'inv_K_cam{}'.format(int(camera))
        matrix = np.array([float(values[i]) for i in range(len(values))])
        K = np.eye(3)
        K[0, 0] = matrix[0]
        K[1, 1] = matrix[1]
        K[0, 2] = matrix[2]
        K[1, 2] = matrix[3]
        item = {
            key: K,
            inv_key: np.linalg.pinv(K)
        }
        data['{}'.format(camera)] = item

    return K

def read_mvsec_extrinsic(extrinsic_fn):
    """
    We assume the extrinsic is obtained by ORBSLAM3,
    The pose of it is from camera to world, but we need world to camera, so we have to inverse it
    """
    lineid = 0
    data = {}
    with open(extrinsic_fn, 'r') as fp:
        for line in fp.readlines():
            values = line.rstrip().split(' ')
            frame = '{:06d}'.format(lineid)
            camera = '02'
            key = 'T_cam{}'.format(camera)
            inv_key = 'inv_T_cam{}'.format(camera)
            matrix = np.array([float(values[i]) for i in range(len(values))])
            matrix = matrix.reshape(3, 4)
            matrix = np.concatenate((matrix, np.array([[0, 0, 0, 1]])), axis=0)
            item = {
                key: matrix,
                inv_key: np.linalg.inv(matrix),
            }
            data['Frame{}:{}'.format(frame, camera)] = item
            lineid += 1

    return data
def getExtrinsic(extrinsics, img_id):
        # 'testing/sequences/000000/image_2/000000.npy'
        # the transformation matrix is from world original point to current frame
        # 4x4
        try: 
            left_T = torch.from_numpy(extrinsics['Frame{:06d}:02'.format(img_id)]['T_cam02'])
        except:
            print(f'Wrong extrinsic data or iamge id')
        left_inv_T = torch.from_numpy( extrinsics['Frame{:06d}:02'.format(img_id)]['inv_T_cam02'])

        return left_T, left_inv_T
def mesh_grid(b, h, w, device, dtype=torch.float):
    """ construct pixel coordination in an image"""
    # [1, H, W]  copy 0-width for h times  : x coord
    x_range = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    # [1, H, W]  copy 0-height for w times : y coord
    y_range = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)

    # [b, 2, h, w]
    pixel_coord = torch.cat((x_range, y_range), dim=1)

    return pixel_coord
def project_to_3d(depth, K, inv_K=None, T_target_to_source:torch.Tensor=None, eps=1e-7):
    """
    project depth map to 3D space
    Args:
        depth:                          (Tensor): depth map(s), can be several depth maps concatenated at channel dimension
                                        [BatchSize, Channel, Height, Width]
        K:                              (Tensor): instrincs of camera
                                        [BatchSize, 3, 3] or [BatchSize, 4, 4]
        inv_K:                          (Optional, Tensor): invserse instrincs of camera
                                        [BatchSize, 3, 3] or [BatchSize, 4, 4]
        T_target_to_source:             (Optional, Tensor): predicted transformation matrix from target image to source image frames
                                        [BatchSize, 4, 4]
        eps:                            (float): eplison value to avoid divide 0, default 1e-7

    Returns: Dict including
        homo_points_3d:                 (Tensor): the homogeneous points after depth project to 3D space, [x, y, z, 1]
                                        [BatchSize, 4, Channel*Height*Width]
    if T_target_to_source provided:

        triangular_depth:               (Tensor): the depth map after the 3D points project to source camera
                                        [BatchSize, Channel, Height, Width]
        optical_flow:                   (Tensor): by 3D projection, the rigid flow can be got
                                        [BatchSize, Channel*2, Height, Width], to get the 2nd flow, index like [:, 2:4, :, :]
        flow_mask:                      (Tensor): the mask indicates which pixel's optical flow is valid
                                        [BatchSize, Channel, Height, Width]
    """

    # support C >=1, for C > 1, it means several depth maps are concatenated at channel dimension
    B, C, H, W = depth.size()
    device = depth.device
    dtype = depth.dtype
    output = {}

    # [B, 2, H, W]
    pixel_coord = mesh_grid(B, H, W, device, dtype)
    ones = torch.ones(B, 1, H, W, device=device, dtype=dtype)
    # [B, 3, H, W], homogeneous coordination of image pixel, [x, y, 1]
    homo_pixel_coord = torch.cat((pixel_coord, ones), dim=1).contiguous()

    # [B, 3, H*W] -> [B, 3, C*H*W]
    homo_pixel_coord = homo_pixel_coord.view(B, 3, -1).repeat(1, 1, C).contiguous()
    # [B, C*H*W] -> [B, 1, C*H*W]
    depth = depth.view(B, -1).unsqueeze(dim=1).contiguous()
    if inv_K is None:
        inv_K = torch.inverse(K[:, :3, :3])
    # [B, 3, C*H*W]
    points_3d = torch.matmul(inv_K[:, :3, :3], homo_pixel_coord) * depth
    ones = torch.ones(B, 1, C*H*W, device=device, dtype=dtype)
    # [B, 4, C*H*W], homogeneous coordiate, [x, y, z, 1]
    homo_points_3d = torch.cat((points_3d, ones), dim=1)
    output['homo_points_3d'] = homo_points_3d

    if T_target_to_source is not None:
        if K.shape[-1] == 3:
            new_K = torch.eye(4, device=device, dtype=dtype).unsqueeze(dim=0).repeat(B, 1, 1)
            new_K[:, :3, :3] = K[:, :3, :3]
            # [B, 3, 4]
            P = torch.matmul(new_K, T_target_to_source)[:, :3, :]
        else:
            # [B, 3, 4]
            P = torch.matmul(K, T_target_to_source)[:, :3, :]
        # [B, 3, C*H*W]
        src_points_3d = torch.matmul(P, homo_points_3d)

        # [B, C*H*W] -> [B, C, H, W], the depth map after 3D points projected to source camera
        triangular_depth = src_points_3d[:, -1, :].reshape(B, C, H, W).contiguous()
        output['triangular_depth'] = triangular_depth
        # [B, 2, C*H*W]
        src_pixel_coord = src_points_3d[:, :2, :] / (src_points_3d[:, 2:3, :] + eps)
        # [B, 2, C, H, W] -> [B, C, 2, H, W]
        src_pixel_coord = src_pixel_coord.reshape(B, 2, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # [B, C, 1, H, W]
        mask = (src_pixel_coord[:, :, 0:1] >=0) & (src_pixel_coord[:, :, 0:1] <= W-1) \
               & (src_pixel_coord[:, :, 1:2] >=0) & (src_pixel_coord[:, :, 1:2] <= H-1)

        # valid flow mask
        mask = mask.reshape(B, C, H, W).contiguous()
        output['flow_mask'] = mask
        # [B, C*2, H, W]
        src_pixel_coord = src_pixel_coord.reshape(B, C*2, H, W).contiguous()
        output['src_pixel_coord'] = src_pixel_coord
        # [B, C*2, H, W]
        optical_flow = src_pixel_coord - pixel_coord.repeat(1, C, 1, 1)
        output['optical_flow'] = optical_flow

    return output
if __name__ == '__main__':
    sequence_path = './dataset/indoor_flying_1'
    
    
    disp_dir_path = os.path.join(sequence_path, 'disparity_image')
    left_dir_path = os.path.join(sequence_path, 'image0')
    right_dir_path = os.path.join(sequence_path, 'image1')
    calib_file_path = 'dataset/calib/camchain-imucam-indoor_flying.yaml'
    odom_file_path = os.path.join(sequence_path, 'odometry.txt')

    i = 800
    gap = 3
    # disp_file_path = os.path.join(disp_dir_path, png_name(i))
    image1_file_path = os.path.join(left_dir_path, png_name(i))
    image2_file_path = os.path.join(left_dir_path, png_name(i+gap))
    # print(disp_file_path)
    # disp = cv2.imread(disp_file_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(image1_file_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_file_path, cv2.IMREAD_GRAYSCALE)
    K = read_mvsec_projection(calib_file_path)
    # K = read_mvsec_intrinsic(calib_file_path)
    odometry_data = read_mvsec_extrinsic(odom_file_path)
    T1, T1_inv = getExtrinsic(odometry_data, i)
    T2, T2_inv = getExtrinsic(odometry_data, i+gap)
    T_past_to_now = torch.matmul(T2, T1_inv)

    h, w = img1.shape[:2]
    img1_tensor = torch.tensor(img1, dtype=torch.float64).view(1, 1, h, w)
    i1, i2 = K.shape[-2:]
    K = torch.tensor(K).view(1, i1, i2)
    T_past_to_now = T_past_to_now.view(1, 4, 4)
    output = project_to_3d(img1_tensor, K, inv_K=None, T_target_to_source=T_past_to_now)
    src_coord = output['src_pixel_coord'].squeeze(dim=0).numpy().astype('float32')
    flow = output['optical_flow'].squeeze(dim=0).permute(1, 2, 0).numpy().astype('float32')
    map2, map1 = np.indices((h, w), dtype=np.float32)
    # map1 = map1 + disp
    # breakpoint()
    warp = cv2.remap(img1, src_coord[0], src_coord[1], cv2.INTER_LINEAR)
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=True)
    zeros = np.zeros_like(warp)
    show_warped = np.stack((img1, zeros, warp), axis = 2)
    show_diff = np.stack((img1, zeros, img2), axis = 2)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    cv2.imshow('warped', show_warped)
    cv2.imshow('diff', show_diff)
    # cv2.imshow('flow', flow_color)
    cv2.waitKey(0)