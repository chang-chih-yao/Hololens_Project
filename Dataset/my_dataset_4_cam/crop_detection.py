import numpy as np
import os
import cv2
import tensorflow as tf
import scipy.io
from object_detection.utils import label_map_util
from shutil import copyfile

import time

#file_arr = ['D:/Dataset/Action/my_dataset_4_cam/1', 'D:/Dataset/Action/my_dataset_4_cam/2', 'D:/Dataset/Action/my_dataset_4_cam/3', 'D:/Dataset/Action/my_dataset_4_cam/4', 'D:/Dataset/Action/my_dataset_4_cam/5', 'D:/Dataset/Action/my_dataset_4_cam/6']
file_arr = ['D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/1', 
            'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/2', 
            'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/3', 
            'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/4', 
            'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/5', 
            'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/6']

old_file_num = [208, 285, 320, 309, 280, 316]              # [ 舊的 label_1 資料夾的個數, 舊的 label_2 資料夾的個數, ... ] 這樣他就會跳過舊的，只crop新的片段

def object_detection(detection_graph, category_index):
    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=detection_graph, config=config) as sess:

            fps = 0
            human = 0
            total_err = 1

            for arr in range(len(file_arr)):
                err = 0
                for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
                    for f in fileNames:
                        img_path = os.path.join(dirPath, f)
                        img_path = img_path.replace('\\', '/')                                                   # 'D:/Dataset/Action/my_dataset_4_cam/1/0/0001/img_00001.jpg'
                        img_dir = img_path.split('img_')[0].replace('my_dataset_4_cam', 'my_dataset_4_cam/crop') # 'D:/Dataset/Action/my_dataset_4_cam/crop/1/0/0001/'
                        img_name = img_dir + img_path.split('/')[-1]                                             # 'D:/Dataset/Action/my_dataset_4_cam/crop/1/0/0001/img_00001.jpg'
                        
                        if int(img_dir.split('/')[-2]) <= old_file_num[arr]:
                            break

                        if not os.path.exists(img_dir):
                            os.mkdir(img_dir)
                        if not os.path.exists(img_path):
                            print('path error!')
                            exit()
                            
                        frame_0 = cv2.imread(img_path)
                        crop_img = None

                        image_np_expanded = np.expand_dims(frame_0, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                        boxes_new = np.squeeze(boxes)
                        classes_new = np.squeeze(classes).astype(np.int32)
                        scores_new = np.squeeze(scores)
                        category_index_new = category_index
                        max_boxes_to_draw = 4
                        min_score_thresh = .7
                        max_people = 1

                        for i in range(min(max_boxes_to_draw, boxes_new.shape[0])):
                            if scores_new is None or (scores_new[i] > min_score_thresh):
                                test1 = None

                                if category_index_new[classes_new[i]]['name']:
                                    test1 = category_index_new[classes_new[i]]['name']

                                #   we only do which detection class is person
                                if test1 == "person":
                                    if max_people == 0:
                                        break
                                    max_people -= 1

                                    human = 1
                                    test_box = boxes_new[i]

                                    #   get detection's left, right, top, bottom
                                    height, width = frame_0.shape[:2]
                                    #print(str(height), str(width))
                                    (left, right, top, bottom) = (int(test_box[1] * width), int(test_box[3] * width), int(test_box[0] * height), int(test_box[2] * height))
                                    # (left, top, width, height) = (left, top, right - left, bottom - top)
                                    # frame_0 = cv2.rectangle(frame_0, (left, top), (right, bottom), (0, 255, 0), 2)
                                    if top - 5 < 0:
                                        top = 0
                                    else:
                                        top -= 5

                                    if bottom + 5 >= height:
                                        bottom = height - 1
                                    else:
                                        bottom += 5

                                    height = bottom - top + 1

                                    mid = (left + right)/2

                                    if (mid - (height/2)) < 0:
                                        left = 0
                                    else:
                                        left = mid - (height/2)

                                    if (mid + (height/2)) >= width:
                                        right = width - 1
                                    else:
                                        right = mid + (height/2)
                                    # print(str(top), str(bottom), str(left), str(right))
                                    crop_img = frame_0[top:bottom, int(left):int(right)].copy()
                                    crop_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                        if human == 1:
                            human = 0
                            cv2.imwrite(img_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            #print(img_name)
                            cv2.imshow('crop', crop_img)
                            cv2.putText(frame_0, "FPS: %.2f" % (1.0 / (time.time() - fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame_0, img_path, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.imshow("origin", frame_0)
                            if cv2.waitKey(1) == 27:
                                exit()
                        else:
                            print(img_name)
                            crop_img = frame_0[142:398, 352:608]
                            cv2.imwrite(img_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            copyfile(img_path, 'D:/Dataset/Action/my_dataset_4_cam/err' + '/img_{:05d}.jpg'.format(total_err))
                            cv2.imshow('crop', crop_img)
                            cv2.putText(frame_0, "FPS: %.2f" % (1.0 / (time.time() - fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame_0, img_path, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow("origin", frame_0)
                            if cv2.waitKey(1) == 27:
                                exit()
                            err += 1
                            total_err += 1
                        
                        fps = time.time()
                print('-------------------------------------')
                print('err:', err)
                print('-------------------------------------')
                # print(frame)

    cv2.destroyAllWindows()


def main():
    # cam_num = 4
    #video_root = 'D:/code/tf-openpose_new/one_million.mp4'
    #video_root = 'D:/code/tf-openpose_new/baseball_video/baseball_1.mp4'
    # save_root = 'D:/Code/MultiCamOverlap/dataset/detections/No3/'
    # MODEL_NAME
    MODEL_NAME = 'D:/code/Tensorflow/weight/faster_rcnn_inception_v2_coco_2018_01_28'
    #MODEL_NAME = 'D:/code/Tensorflow/weight/ssd_mobilenet_v2_coco_2018_03_29'
    # Path to frozen detection graph. This is the actual model
    # - that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    #PATH_TO_LABELS = os.path.join('data_object', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = 'D:/code/Tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
    NUM_CLASSES = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    object_detection(detection_graph, category_index)


if __name__ == '__main__':
    main()
