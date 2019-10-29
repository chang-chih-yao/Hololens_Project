import os
import cv2
import numpy as np
import shutil

root_path = '11_action_15_class_MOD_5/'

with open(root_path + 'result.txt', 'r') as f:
    Lines = f.readlines()
    

print(len(Lines))

img_index = 2

while(True):
    img1 = cv2.imread(root_path + '{:04d}.jpg'.format(img_index-2))
    img2 = cv2.imread(root_path + '{:04d}.jpg'.format(img_index-1))
    img3 = cv2.imread(root_path + '{:04d}.jpg'.format(img_index))
    vis = np.concatenate((img1, img2, img3), axis=1)
    cv2.putText(vis, Lines[img_index].replace('\n', ''), (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('result', vis)
    key = cv2.waitKey(20)
    if key == 27:
        break
    elif key == 97:       # a
        if img_index > 2:
            img_index -= 1
    elif key == 100:      # d
        if img_index < len(Lines)-1:
            img_index += 1