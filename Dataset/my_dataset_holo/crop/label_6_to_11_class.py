import cv2
import os
import numpy as np

file_root = ['2/', '3/', '4/', '5/', '6/']

def main():
    for i in range(len(file_root)):
        print(len(os.listdir(file_root[i])))

if __name__ == "__main__":
    main()