import cv2
import numpy as np
import os
import sys


img = cv2.imread(sys.argv[1])


def click(event, x,y,s,p):
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x-126
        x2 = x+126
        new_img = img[:, x1:x2, :].copy()
        cv2.imwrite('out.jpg', new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        print('img saved')

cv2.namedWindow('crop')
cv2.setMouseCallback('crop', click)




while(True):
    cv2.imshow('crop',img)
    if cv2.waitKey(10) == 27:
        break