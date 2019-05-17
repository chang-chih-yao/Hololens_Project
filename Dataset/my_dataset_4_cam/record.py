import cv2
import numpy as np
import os
import time
import threading
'''
1920 1080
1600 900
1366 768
1280 720
1024 576
960 540
640 360
'''

class ipcamCapture:
    def __init__(self, camera_ID):
        self.Frame = None
        self.status = False
        self.isstop = False
        self.flag = False
		
        self.capture = cv2.VideoCapture(camera_ID)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

    def start(self):
        print('ipcam started!')
        self.t = threading.Thread(target=self.queryframe, daemon=True, args=())
        self.t.start()

    def stop(self):
        self.isstop = True
        while(not self.flag):
            time.sleep(0.1)
        #print('ipcam stopped!')
   
    def getframe(self):
        return self.status, self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
            time.sleep(0.01)
        print('close')
        self.capture.release()
        self.flag = True

def record(file_root):

    print('folder name:')
    my_str = input()
    if not os.path.exists(file_root + my_str + '/'):
        os.mkdir(file_root + my_str + '/')
        os.mkdir(file_root + my_str + '/0/')
        os.mkdir(file_root + my_str + '/1/')
        os.mkdir(file_root + my_str + '/2/')
        os.mkdir(file_root + my_str + '/3/')
    else:
        print('folder existed, continue?(y/n)')
        zz = input()
        if zz != 'y':
            exit()

    cap_0 = ipcamCapture(cv2.CAP_DSHOW + 0)
    cap_0.start()
    cap_1 = ipcamCapture(cv2.CAP_DSHOW + 1)
    cap_1.start()
    cap_2 = ipcamCapture(cv2.CAP_DSHOW + 2)
    cap_2.start()
    cap_3 = ipcamCapture(cv2.CAP_DSHOW + 3)
    cap_3.start()

    cv2.namedWindow('result 0')
    cv2.moveWindow('result 0', 0, 0)
    cv2.namedWindow('result 1')
    cv2.moveWindow('result 1', 960, 0)
    cv2.namedWindow('result 2')
    cv2.moveWindow('result 2', 0, 540)
    cv2.namedWindow('result 3')
    cv2.moveWindow('result 3', 960, 540)

    cou = 1
    fps_time = 0

    while(True):
        ret_flag, frame_ = cap_3.getframe()
        if ret_flag == False:
            time.sleep(0.2)
            continue
        else:
            break

    while(True):
        
        ret_0, frame_0 = cap_0.getframe()
        ret_1, frame_1 = cap_1.getframe()
        ret_2, frame_2 = cap_2.getframe()
        ret_3, frame_3 = cap_3.getframe()

        img_dir_0 = file_root + my_str + '/0/img_{:05d}.jpg'.format(cou)
        img_dir_1 = file_root + my_str + '/1/img_{:05d}.jpg'.format(cou)
        img_dir_2 = file_root + my_str + '/2/img_{:05d}.jpg'.format(cou)
        img_dir_3 = file_root + my_str + '/3/img_{:05d}.jpg'.format(cou)

        cv2.imwrite(img_dir_0, frame_0, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        cv2.imwrite(img_dir_1, frame_1, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        cv2.imwrite(img_dir_2, frame_2, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        cv2.imwrite(img_dir_3, frame_3, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        cv2.putText(frame_0, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('result 3', frame_3)
        cv2.imshow('result 2', frame_2)
        cv2.imshow('result 1', frame_1)
        cv2.imshow('result 0', frame_0)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap_0.stop()
            cap_1.stop()
            cap_2.stop()
            cap_3.stop()
            break

        fps_time = time.time()
        cou += 1



if __name__ == '__main__':
    
    record(file_root = 'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/Raw_data/')