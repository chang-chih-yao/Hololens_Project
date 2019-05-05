import cv2
import time
import os
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

w, h = model_wh('432x368')
# e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))
fps_time = 0

file_arr = ['D:/Dataset/Action/my_dataset/1', 'D:/Dataset/Action/my_dataset/2']  # 如果有新的action，要調整
old_file_num = [83, 87]              # [ 舊的 label_1 資料夾的個數, 舊的 label_2 資料夾的個數 ] 這樣他就會跳過舊的，只crop新的片段

for arr in range(2):
    err = 0
    for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
        for f in fileNames:
            img_path = os.path.join(dirPath, f)
            img_path = img_path.replace('\\', '/')                                       # 'D:/Dataset/Action/my_dataset/1/1_0001/img_00001.jpg'

            img_dir = img_path.split('img_')[0].replace('my_dataset', 'my_dataset/crop') # 'D:/Dataset/Action/my_dataset/crop/1/1_0001/'
            img_name = img_dir + img_path.split('/')[-1]                                 # 'D:/Dataset/Action/my_dataset/crop/1/1_0001/img_00001.jpg'
            # print(img_name)
            if int(img_dir.split('_')[-1].replace('/', '')) <= old_file_num[arr]:
                break
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            
            img = cv2.imread(img_path)
            image = img[:, 30:410, :].copy()

            humans = e.inference(image)
            temp_class = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            cv2.putText(temp_class.npimg,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', temp_class.npimg)

            fps_time = time.time()
            # print(temp_class.Neck_x, temp_class.Neck_y)
            if temp_class.Neck_x == 0 or temp_class.Neck_y == 0:
                err += 1
                print(img_name)
            x1 = temp_class.Neck_x + 30 - 126
            x2 = temp_class.Neck_x + 30 + 126

            crop_img = img[:, x1:x2, :]
            cv2.imwrite(img_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            cv2.waitKey(1)
    print('err:', err)



# for background
for i in range(42):
    img = cv2.imread('D:/Dataset/Action/my_dataset/2/2_0073/img_{:05d}.jpg'.format(i+1))
    image = img[:, 224-126:224+126, :].copy()
    cv2.imwrite('D:/Dataset/Action/my_dataset/crop/2/2_0073/img_{:05d}.jpg'.format(i+1), image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
