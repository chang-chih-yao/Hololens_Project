import cv2
import time
import os
import numpy as np
import json

import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

class MyCustomEncoder(json.JSONEncoder):
    def iterencode(self, obj):
        if isinstance(obj, float):
            yield format(obj, '.3f')
        elif isinstance(obj, dict):
            last_index = len(obj) - 1
            yield '{'
            i = 0
            for key, value in obj.items():
                yield '"' + key + '": '
                for chunk in MyCustomEncoder.iterencode(self, value):
                    yield chunk
                if i != last_index:
                    yield ", "
                i+=1
            yield '}'
        elif isinstance(obj, list):
            last_index = len(obj) - 1
            yield "["
            for i, o in enumerate(obj):
                for chunk in MyCustomEncoder.iterencode(self, o):
                    yield chunk
                if i != last_index: 
                    yield ", "
            yield "]"
        else:
            for chunk in json.JSONEncoder.iterencode(self, obj):
                yield chunk

################################### OpenPose ###########################################
gpu_options = tf.GPUOptions(allow_growth=True)
#e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
#e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))

fps_time = 0
'''
file_arr = ['D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/1',
            'D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/2_start',
            'D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/2_end',
            'D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/3',
            'D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/4',
            'D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/5',
            'D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/6']  # 如果有新的action，要調整
'''
file_arr = ['D:/Code/Hololens_Project/Dataset/my_dataset_holo/crop/1']
old_file_num = [217, 215, 139, 137, 130, 131]                      # [ 舊的 label_1 資料夾的個數, 舊的 label_2 資料夾的個數, ... ] 這樣他就會跳過舊的，只crop新的片段

label_name = {1:'1', 2:'2_start', 3:'2_end', 4:'3', 5:'4', 6:'5', 7:'6'}

for arr in range(len(file_arr)):
    err = 0
    folder_cou = 1
    for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
        if len(dirPath.replace('\\', '/').split('/')) == 8:
            #print(dirPath)
            output = {'data':[], 'label':label_name[arr+1], 'label_index':arr+1}
            json_name = 'Null'
            for f in fileNames:
                img_path = os.path.join(dirPath, f)
                img_path = img_path.replace('\\', '/')                                   # 'D:/Code/Hololens_Project/Dataset/my_dataset_holo/1/1_0001/img_00001.jpg'
                img_frame_num = int(img_path.split('_')[-1].split('.')[0])
                json_name = img_path.split('/')[-3]

                #img_dir = img_path.split('img_')[0].replace('my_dataset_holo', 'my_dataset_holo/crop') # 'D:/Dataset/Action/my_dataset/crop/1/1_0001/'
                #img_name = img_dir + img_path.split('/')[-1]                                           # 'D:/Dataset/Action/my_dataset/crop/1/1_0001/img_00001.jpg'
                
                # print(img_name)
                # if int(img_dir.split('_')[-1].replace('/', '')) <= old_file_num[arr]:
                #     break
                '''
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)
                '''
                image = cv2.imread(img_path)
                cv2.imshow('origin', image)
                #image = img[:, 30:410, :].copy()

                pose = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                score = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

                humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
                if len(humans) == 0:       # no human
                    err += 1
                    print(img_path)
                    frame = {'frame_index':img_frame_num, 'skeleton':[]}
                    output['data'].append(frame)
                elif humans[0].score > 0.7:
                    #print(humans[0].score)
                    #print(len(humans[0].body_parts))
                    for _, item in humans[0].body_parts.items():
                        #print(item.part_idx)
                        pose[int(item.part_idx)] = item.x
                        pose[int(item.part_idx)*2+1] = item.y
                        score[int(item.part_idx)] = item.score
                        #print('%.3f %.3f %.3f' % (humans[0].body_parts[i].x, humans[0].body_parts[i].y, humans[0].body_parts[i].score))
                    
                    frame = {'frame_index':img_frame_num, 'skeleton':[{'pose':pose, 'score':score}]}
                    output['data'].append(frame)
                    npimg, key_points = TfPoseEstimator.draw_one_human(image, humans, imgcopy=False, score=0.8)

                    cv2.putText(npimg,  "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('tf-pose-estimation result', npimg)

                    fps_time = time.time()

                # print(temp_class.Neck_x, temp_class.Neck_y)
                    
                '''
                x1 = key_points[1][0] + 30 - 126
                x2 = key_points[1][0] + 30 + 126

                crop_img = img[:, x1:x2, :]
                cv2.imwrite(img_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                '''
                cv2.waitKey(1)
            with open('skeleton_json/{}_{:04d}.json'.format(str(arr+1), folder_cou), 'w') as f:
                json.dump(output, f, cls=MyCustomEncoder)
            folder_cou += 1
    print('err:', err)



# for background
'''
for i in range(42):
    img = cv2.imread('D:/Dataset/Action/my_dataset/1/1_0136/img_{:05d}.jpg'.format(i+1))
    image = img[:, 224-126:224+126, :].copy()
    cv2.imwrite('D:/Dataset/Action/my_dataset/crop/1/1_0136/img_{:05d}.jpg'.format(i+1), image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
'''
