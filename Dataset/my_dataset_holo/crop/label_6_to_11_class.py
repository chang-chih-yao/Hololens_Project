import cv2
import os
import numpy as np

'''
generate "for_11_class.txt"
'''

#file_arr = ['2/', '3/', '4/', '5/', '6/']
file_arr = ['2/', '3/']

def main():
    for i in range(len(file_arr)):
        print(len(os.listdir(file_arr[i])))
        for dirPath, dirNames, fileNames in os.walk(file_arr[i]):
            print(dirPath, fileNames)

if __name__ == "__main__":
    main()