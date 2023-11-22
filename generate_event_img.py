import sys
import os
import numpy as np
import cv2
import yaml

def png_name(i: int): 
    file_name = '%06d.png' % i
    return file_name

if __name__ == '__main__':
    sequence_path = './dataset/indoor_flying_3'
    
    disp_dir_path = os.path.join(sequence_path, 'disparity_image')
    left_dir_path = os.path.join(sequence_path, 'image0')
    right_dir_path = os.path.join(sequence_path, 'image1')
    calib_file_path = 'dataset/calib/camchain-imucam-indoor_flying.yaml'
    odom_file_path = os.path.join(sequence_path, 'odometry.txt')
    left_event_path = os.path.join(sequence_path, 'event0')
    assert os.path.exists(left_event_path)
    left_event_file = os.listdir(left_event_path)
    left_event_file.sort()
    for event_file_name in left_event_file:
        if '.npy' not in event_file_name:
            continue
        full_path = os.path.join(left_event_path, event_file_name)
        event = np.load(full_path)
        event_img = np.zeros((260, 346, 3))
        
        for i in range(len(event)):
            e = event[i]
            if e[-1] > 0:
                event_img[int(e[2]), int(e[1]), 2] = 255
            else:
                event_img[int(e[2]), int(e[1]), 0] = 255
        
        event_image_path = os.path.join(left_event_path, event_file_name.split('.')[0]+".png")
        cv2.imwrite(event_image_path, event_img)
        # break