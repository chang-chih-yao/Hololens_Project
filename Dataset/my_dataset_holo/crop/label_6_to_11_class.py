import cv2
import os
import numpy as np
from shutil import copyfile

'''
generate "for_11_class.txt"
'''

file_arr = ['2/', '3/', '4/', '5/', '6/']
#file_arr = ['2/', '3/']
file_arr_start = ['2_start/', '3_start/', '4_start/', '5_start/', '6_start/']
file_arr_end = ['2_end/', '3_end/', '4_end/', '5_end/', '6_end/']

def main():
    for i in range(len(file_arr)):
        #print(len(os.listdir(file_arr[i])))
        start_cou = 1
        end_cou = 1
        for dirPath, dirNames, fileNames in os.walk(file_arr[i]):
            dirPath = dirPath.replace('\\', '/')
            print(dirPath)
            if len(fileNames) == 0:
                continue

            img_cou = 0
            while(True):
                if img_cou >= len(fileNames):
                    if not os.path.exists(file_arr_start[i] + dirPath.split('/')[-1]):
                        os.mkdir(file_arr_start[i] + file_arr[i][0] + '_{:04d}'.format(start_cou))
                    for s in range(len(fileNames)):
                        copyfile(dirPath + '/' + fileNames[s], file_arr_start[i] + file_arr[i][0] + '_{:04d}'.format(start_cou) + '/img_{:05d}.jpg'.format(s+1))
                    start_cou += 1
                    break
                img_path = dirPath + '/' + fileNames[img_cou]
                img = cv2.imread(img_path)
                cv2.putText(img, img_path, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('result', img)
                key = cv2.waitKey(20)
                if key == 27:
                    cv2.destroyAllWindows()
                    exit()
                elif key == 104:               # h
                    img_cou += 1
                elif key == 102:               # f
                    if img_cou != 0:
                        img_cou -= 1
                elif key == 32:                # space
                    if not os.path.exists(file_arr_start[i] + dirPath.split('/')[-1]):
                        os.mkdir(file_arr_start[i] + file_arr[i][0] + '_{:04d}'.format(start_cou))
                    if not os.path.exists(file_arr_end[i] + dirPath.split('/')[-1]):
                        os.mkdir(file_arr_end[i] + file_arr[i][0] + '_{:04d}'.format(end_cou))
                    
                    for s in range(img_cou):
                        copyfile(dirPath + '/' + fileNames[s], file_arr_start[i] + file_arr[i][0] + '_{:04d}'.format(start_cou) + '/img_{:05d}.jpg'.format(s+1))
                    for e in range(img_cou, len(fileNames)):
                        copyfile(dirPath + '/' + fileNames[e], file_arr_end[i] + file_arr[i][0] + '_{:04d}'.format(end_cou) + '/img_{:05d}.jpg'.format(e-img_cou+1))
                    
                    start_cou += 1
                    end_cou += 1
                    break

if __name__ == "__main__":
    main()