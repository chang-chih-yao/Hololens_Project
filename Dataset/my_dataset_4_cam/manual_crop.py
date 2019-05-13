import cv2
import numpy as np
import os
import sys


with open('err.txt', 'r') as file:
    lines = file.read().splitlines()

print(lines[0])
print(len(lines))



cropping = False
cou = 0
 
x_start, y_start, x_end, y_end = 0, 0, 0, 0

oriImage = None

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, cou
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            height = y_end - y_start + 1
            mid = (x_start + x_end)/2
            left = mid - (height/2)
            right = mid + (height/2)
            roi = oriImage[y_start:y_end, int(left):int(right)]
            roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(lines[cou], roi, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            cv2.imshow("Cropped", roi)
            if cou + 1 < 1792:
                cou += 1
            else:
                print('end')
                exit()
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while True:
    image = cv2.imread(lines[cou].replace('crop/', ''))
    oriImage = image.copy()
 
    i = image.copy()
 
    if not cropping:
        cv2.putText(image, lines[cou], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(cou), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("image", image)
 
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
 
    if cv2.waitKey(10) == 27:
        break
    elif cv2.waitKey(10) == 104:
        cou += 1
    elif cv2.waitKey(10) == 102:
        cou -= 1
 
# close all open windows
cv2.destroyAllWindows()


'''
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
'''