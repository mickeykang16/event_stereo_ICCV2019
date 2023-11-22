import sys
import os
import numpy as np
import cv2

def png_name(i: int): 
    file_name = '%06d.png' % i
    return file_name

if __name__ == '__main__':
    sequence_path = './dataset_1x/indoor_flying_1'
    
    
    disp_dir_path = os.path.join(sequence_path, 'disparity_image')
    left_dir_path = os.path.join(sequence_path, 'image0')
    right_dir_path = os.path.join(sequence_path, 'image1')

    i = 200
    disp_file_path = os.path.join(disp_dir_path, png_name(i))
    left_file_path = os.path.join(left_dir_path, png_name(i))
    right_file_path = os.path.join(right_dir_path, png_name(i))
    # print(disp_file_path)
    disp = cv2.imread(disp_file_path, cv2.IMREAD_GRAYSCALE)
    left = cv2.imread(left_file_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_file_path, cv2.IMREAD_GRAYSCALE)
    h, w = disp.shape[:2]
    map2, map1 = np.indices((h, w), dtype=np.float32)
    map1 = map1 + disp
    left_remap = cv2.remap(left, map1, map2, cv2.INTER_LINEAR)

    zeros = np.zeros_like(left_remap)
    show_lr = np.stack((right, zeros, left_remap), axis = 2)
    import pdb; pdb.set_trace()
    cv2.imshow('disp', disp)
    cv2.imshow('left_remap', left_remap)
    cv2.imshow('right', right)
    cv2.imshow('diff', show_lr)
    cv2.waitKey(0)