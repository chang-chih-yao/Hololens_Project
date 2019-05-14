import numpy as np
import os
import cv2
import time
import socket
import scipy.io
import threading
import math

import tensorflow as tf
from object_detection.utils import label_map_util

import torch
import torchvision
import torch.backends.cudnn as cudnn

from tsn_pytorch.models import TSN
from tsn_pytorch.transforms import *

'''
python detection.py --video_root D:/Code/AF_tracking/videos/ --save_root D:/Code/AF_tracking/dataset/detections/new_delete_other/ --cam_num 4
'''

start_time = [1, 1, 1, 1]
start_sequence = 0
end_sequence = 810
HOST = '192.168.11.107'
PORT = 9000
num_class = 21


class ipcamCapture:
    def __init__(self, camera_ID):
        self.Frame = None
        self.status = False
        self.isstop = False
        self.flag = False
		
        self.capture = cv2.VideoCapture(camera_ID)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

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
        print('close')
        self.capture.release()
        self.flag = True

def is_num(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def validate(model, rst):

    input_var = torch.autograd.Variable(rst, volatile=True)
    output = model(input_var)

    topk = (1,)
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    top1_val = int(pred[0][0])
    if top1_val == 0:
        top1_val = num_class
    
    return top1_val


def object_detection(detection_graph, video_root, category_index, register):

    print('regist_shape : ' + str(register.shape))

    ############################## Action Recognition ######################################
    model = TSN(num_class, 3, 'RGB', base_model='resnet34', consensus_type='avg', dropout=0.7)

    checkpoint = torch.load('C:\\Users\\ray20\\Desktop\\Michael\\Code\\Action_Recognition\\tsn_pytorch\\pth\\4cam_2019_0425_6_actions.pth')
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)

    crop_size = model.crop_size

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # switch to evaluate mode
    model.eval()

    trans = trans = torchvision.transforms.Compose([
        GroupScale(int(crop_size)),
        Stack(roll=False),
        ToTorchFormatTensor(div=True)
        ]
    )

    ############################## Socket ######################################
    print('connecting...')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print('start')

    sock.send(b'4cam')
    sock.recv(1024)

    ############################## VGG16 feature ######################################
    img_to_tensor = torchvision.transforms.ToTensor()
    pytorch_model = torchvision.models.vgg16(pretrained=True).features[:]
    print(pytorch_model)
    pytorch_model.cuda()
    pytorch_model.eval()

    ############################## Human Detection ######################################
    with detection_graph.as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            num_list = ['0', '1', '2', '3']
            cap_0 = ipcamCapture(cv2.CAP_DSHOW + 0)
            cap_0.start()
            cap_1 = ipcamCapture(cv2.CAP_DSHOW + 1)
            cap_1.start()
            cap_2 = ipcamCapture(cv2.CAP_DSHOW + 2)
            cap_2.start()
            cap_3 = ipcamCapture(cv2.CAP_DSHOW + 3)
            cap_3.start()
            
            caps = []
            caps.append(cap_0)
            caps.append(cap_1)
            caps.append(cap_2)
            caps.append(cap_3)
            cv2.namedWindow('origin 0')
            cv2.moveWindow('origin 0', 0, 0)
            cv2.namedWindow('origin 1')
            cv2.moveWindow('origin 1', 960, 0)
            cv2.namedWindow('origin 2')
            cv2.moveWindow('origin 2', 0, 540)
            cv2.namedWindow('origin 3')
            cv2.moveWindow('origin 3', 960, 540)

            fps = 0
            human = 0
            count = 1
            player_position = [[[-100, -100], [-100, -100]], [[-100, -100], [-100, -100]], [[-100, -100], [-100, -100]], [[-100, -100], [-100, -100]]]
            player_ID = [-1, -1, -1, -1]
            rad = 70
            vote_ID = -1

            images = []
            images.append([])
            images.append([])
            images.append([])
            images.append([])
            for temp in range(4):
                images[temp].append([])
                images[temp].append([])

            actions = []
            actions.append([])
            actions.append([])
            actions.append([])
            actions.append([])
            for temp in range(4):
                actions[temp].append(0)
                actions[temp].append(0)
            

            while(True):
                ret_flag, frame_ = caps[0].getframe()
                if ret_flag == False:
                    time.sleep(0.2)
                    continue
                for cam_id in range(len(caps)):
                    ret, frame = caps[cam_id].getframe()
                    frame_copy = frame.copy()

                    image_np_expanded = np.expand_dims(frame, axis=0)
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

                    people_cou = 0
                    max_people = 2

                    pre_x = 0
                    pre_y = 0

                    for i in range(min(max_boxes_to_draw, boxes_new.shape[0])):
                        if scores_new is None or (scores_new[i] > min_score_thresh):
                            test1 = None

                            if category_index_new[classes_new[i]]['name']:
                                test1 = category_index_new[classes_new[i]]['name']

                            #   we only do which detection class is person
                            if test1 == "person":
                                test_box = boxes_new[i]
                                height, width = frame.shape[:2]
                                (left, right, top, bottom) = (int(test_box[1] * width), int(test_box[3] * width), int(test_box[0] * height), int(test_box[2] * height))
                                center_x = int((left + right)/2)
                                center_y = int((top + bottom)/2)
                                if people_cou == 0:
                                    pre_x = center_x
                                    pre_y = center_y
                                    people_cou += 1
                                else:
                                    if ((center_x - pre_x)**2 + (center_y - pre_y)**2) > rad**2:    # 不能距離太近 避免noise(自己影分身)
                                        people_cou += 1
                                        break

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
                                height, width = frame.shape[:2]
                                (left, right, top, bottom) = (int(test_box[1] * width), int(test_box[3] * width), int(test_box[0] * height), int(test_box[2] * height))
                                # (left, top, width, height) = (left, top, right - left, bottom - top)

                                crop_top = top
                                crop_bottom = bottom
                                crop_left = left
                                crop_right = right

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
                                crop_img = frame_copy[top:bottom, int(left):int(right)].copy()
                                crop_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_LINEAR)

                                center_x = int(mid)
                                center_y = int((top+bottom)/2)

                                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), 5)    # tracking point
                                cv2.circle(frame, (center_x, center_y), rad, (0, 255, 0), 2)

                                ################# nearest tracking #################
                                if player_position[cam_id][0][0] == -100 and player_position[cam_id][0][1] == -100:                            # init player 0
                                    print('-------cam : %d player 0 init-------' %(cam_id))
                                    player_ID[cam_id] = 0
                                    player_position[cam_id][player_ID[cam_id]][0] = center_x
                                    player_position[cam_id][player_ID[cam_id]][1] = center_y
                                elif player_position[cam_id][1][0] == -100 and player_position[cam_id][1][1] == -100 and people_cou == 2:      # init player 1
                                    print('-------cam : %d player 1 init-------' %(cam_id))
                                    player_ID[cam_id] = 1
                                    player_position[cam_id][player_ID[cam_id]][0] = center_x
                                    player_position[cam_id][player_ID[cam_id]][1] = center_y
                                else:
                                    pre_0_x = player_position[cam_id][0][0]
                                    pre_0_y = player_position[cam_id][0][1]
                                    pre_1_x = player_position[cam_id][1][0]
                                    pre_1_y = player_position[cam_id][1][1]

                                    if ((center_x - pre_0_x) ** 2 + (center_y - pre_0_y) ** 2) <= ((center_x - pre_1_x) ** 2 + (center_y - pre_1_y) ** 2):
                                        player_ID[cam_id] = 0
                                        player_position[cam_id][player_ID[cam_id]][0] = center_x
                                        player_position[cam_id][player_ID[cam_id]][1] = center_y
                                    else:
                                        player_ID[cam_id] = 1
                                        player_position[cam_id][player_ID[cam_id]][0] = center_x
                                        player_position[cam_id][player_ID[cam_id]][1] = center_y
                                
                                ################# VGG feature tracking #################
                                tensor = img_to_tensor(crop_img.astype(np.float32))
                                tensor = tensor.resize_(1,3,224,224)
                                tensor = tensor.cuda()
                                result = pytorch_model(torch.autograd.Variable(tensor))
                                result_npy = result.data.cpu().numpy()

                                dist = np.sum((register - result_npy)**2, axis=(1, 2, 3))
                                dist_index = np.argmin(dist)
                                dist_int = int(math.sqrt(dist[dist_index]))

                                if dist_index < int(register.shape[0])/2:
                                    player_ID[cam_id] = 0
                                    player_position[cam_id][player_ID[cam_id]][0] = center_x
                                    player_position[cam_id][player_ID[cam_id]][1] = center_y
                                else:
                                    player_ID[cam_id] = 1
                                    player_position[cam_id][player_ID[cam_id]][0] = center_x
                                    player_position[cam_id][player_ID[cam_id]][1] = center_y



                                # if human == 1:
                                #     human = 0
                                # else:
                                #     crop_img = frame[142:398, 352:608].copy()


                                img_tsn = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                                print('cam:' + str(cam_id) + ' player:' +  str(player_ID[cam_id]) + ' dist_index:' + str(dist_index) + ' | ' + str(dist_int))
                                if len(images[cam_id][player_ID[cam_id]]) == 2:
                                    images[cam_id][player_ID[cam_id]].extend([img_tsn])
                                    # print(len(images[cam_id]))
                                    # print(images[cam_id][0].size)
                                    # print(images[cam_id][1].size)
                                    # print(images[cam_id][2].size)
                                    rst = trans(images[cam_id][player_ID[cam_id]])
                                    rst = torch.unsqueeze(rst, 0)
                                    actions[cam_id][player_ID[cam_id]] = validate(model, rst)
                                    
                                    images[cam_id][player_ID[cam_id]].clear()
                                else:
                                    images[cam_id][player_ID[cam_id]].extend([img_tsn])


                                tran_act = ''
                                if actions[cam_id][player_ID[cam_id]] == 1:
                                    tran_act = '1'
                                elif actions[cam_id][player_ID[cam_id]] == 2 or actions[cam_id][player_ID[cam_id]] == 3 or actions[cam_id][player_ID[cam_id]] == 4 or actions[cam_id][player_ID[cam_id]] == 5:
                                    tran_act = '2'
                                elif actions[cam_id][player_ID[cam_id]] == 6 or actions[cam_id][player_ID[cam_id]] == 7 or actions[cam_id][player_ID[cam_id]] == 8 or actions[cam_id][player_ID[cam_id]] == 9:
                                    tran_act = '3'
                                elif actions[cam_id][player_ID[cam_id]] == 10 or actions[cam_id][player_ID[cam_id]] == 11 or actions[cam_id][player_ID[cam_id]] == 12 or actions[cam_id][player_ID[cam_id]] == 13:
                                    tran_act = '4'
                                elif actions[cam_id][player_ID[cam_id]] == 14 or actions[cam_id][player_ID[cam_id]] == 15 or actions[cam_id][player_ID[cam_id]] == 16 or actions[cam_id][player_ID[cam_id]] == 17:
                                    tran_act = '5'
                                elif actions[cam_id][player_ID[cam_id]] == 18 or actions[cam_id][player_ID[cam_id]] == 19 or actions[cam_id][player_ID[cam_id]] == 20 or actions[cam_id][player_ID[cam_id]] == 21:
                                    tran_act = '6'

                                if player_ID[cam_id] == 0:
                                    frame = cv2.rectangle(frame, (crop_left, crop_top), (crop_right, crop_bottom), (255, 0, 0), 5)
                                    cv2.putText(frame, tran_act, (crop_left, crop_top), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                                elif player_ID[cam_id] == 1:
                                    frame = cv2.rectangle(frame, (crop_left, crop_top), (crop_right, crop_bottom), (0, 0, 255), 5)
                                    cv2.putText(frame, tran_act, (crop_left, crop_top), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                                
                                #cv2.imshow('cam : ' + num_list[cam_id] + ' | player : ' + str(player_ID[cam_id]), crop_img)

                    cv2.imshow('origin ' + num_list[cam_id], frame)
                    if cv2.waitKey(1) == 27:
                        sock.send(b'bye')
                        for cam_id_ in range(len(caps)):
                            caps[cam_id_].stop()
                        cv2.destroyAllWindows()
                        sock.close()
                        exit()

                
                print('(' + str(player_position[0][0][0]), str(player_position[0][0][1]) + ') | (' + str(player_position[0][1][0]), str(player_position[0][1][1]) + ')')
                # camera 0 : (player_0 x ,player_0 y) | (player_1 x, players_1, y)
                byte_str = bytes(str(actions[0][0]), 'ascii')
                sock.send(byte_str)
                print(1/(time.time()-fps))
                fps = time.time()
                count += 1
                sock.recv(1024)


    for cam_id in range(len(caps)):
        caps[cam_id].stop()
    cv2.destroyAllWindows()


def regist(detection_graph, category_index, register_num):
    register = np.array([])
    with detection_graph.as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            num_list = ['0', '1', '2', '3']
            cap_0 = ipcamCapture(cv2.CAP_DSHOW + 0)
            cap_0.start()
            cap_1 = ipcamCapture(cv2.CAP_DSHOW + 1)
            cap_1.start()
            cap_2 = ipcamCapture(cv2.CAP_DSHOW + 2)
            cap_2.start()
            cap_3 = ipcamCapture(cv2.CAP_DSHOW + 3)
            cap_3.start()
            
            caps = []
            caps.append(cap_0)
            caps.append(cap_1)
            caps.append(cap_2)
            caps.append(cap_3)

            cou = 0          # features count
            cap_flag = 0     # 連續擷取4個frame的開關

            img_to_tensor = torchvision.transforms.ToTensor()

            pytorch_model = torchvision.models.vgg16(pretrained=True).features[:]
            print(pytorch_model)
            pytorch_model.cuda()
            pytorch_model.eval()

            while(True):
                ret_flag, frame_ = caps[0].getframe()
                if ret_flag == False:
                    time.sleep(0.2)
                    continue
                for cam_id in range(len(caps)):
                    ret, frame = caps[cam_id].getframe()
                    frame_copy = frame.copy()

                    image_np_expanded = np.expand_dims(frame, axis=0)
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
                    max_boxes_to_draw = 2
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

                                test_box = boxes_new[i]

                                #   get detection's left, right, top, bottom
                                height, width = frame.shape[:2]
                                (left, right, top, bottom) = (int(test_box[1] * width), int(test_box[3] * width), int(test_box[0] * height), int(test_box[2] * height))
                                # (left, top, width, height) = (left, top, right - left, bottom - top)

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
                                crop_img = frame_copy[top:bottom, int(left):int(right)].copy()
                                crop_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                                regist_img = crop_img.copy()

                                tensor = img_to_tensor(crop_img.astype(np.float32))
                                tensor = tensor.resize_(1,3,224,224)
                                tensor = tensor.cuda()
                                result = pytorch_model(torch.autograd.Variable(tensor))
                                result_npy = result.data.cpu().numpy()

                                ################### 開始擷取另外三台camera的frame (cam_0在按下'1'的時候就已經截取了) ################
                                if cam_id == 1 and cap_flag == 1:
                                    register = np.concatenate((register, result_npy), axis=0)
                                    cv2.putText(regist_img, str(cou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.imshow('register 1', regist_img)
                                    cou += 1
                                elif cam_id == 2 and cap_flag == 1:
                                    register = np.concatenate((register, result_npy), axis=0)
                                    cv2.putText(regist_img, str(cou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.imshow('register 2', regist_img)
                                    cou += 1
                                elif cam_id == 3 and cap_flag == 1:
                                    register = np.concatenate((register, result_npy), axis=0)
                                    cv2.putText(regist_img, str(cou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.imshow('register 3', regist_img)
                                    cou += 1
                                    cap_flag = 0

                                # 如果已經達到截取數量，就開始比對目前的影像，最像哪一張註冊的frame，並且會print出 L2 dist 跟 index
                                if cou >= register_num:
                                    t1 = time.time()
                                    dist = np.sum((register - result_npy)**2, axis=(1, 2, 3))
                                    dist_int = int(math.sqrt(dist[np.argmin(dist)]))
                                    t2 = time.time()
                                    print(str(dist_int) + ' | ' + str(t2-t1))
                                    cv2.putText(crop_img, str(dist_int), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                                    #print(np.argmin(dist))
                                # else:
                                #     print(result_npy.shape)

                                # 如果目前沒有在擷取，就顯示Ready，告訴使用者隨時可以按'1'，開始截取(註冊)
                                if cap_flag == 0 and cou < register_num:
                                    cv2.putText(crop_img, 'Ready', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                                cv2.imshow('crop ' + str(num_list[cam_id]), crop_img)


                                key = cv2.waitKey(10)
                                if key == 113:                               # press 'q'
                                    if len(register) != 0:
                                        np.save('feature.npy', register)     # save features array
                                        for cam_id_ in range(len(caps)):
                                            caps[cam_id_].stop()
                                        cv2.destroyAllWindows()
                                        return register
                                elif key == 49 and cam_id == 0 and cap_flag == 0:    # press '1' -> 從cam_0連續擷取4個frames，一直到cam_3(會跑4個for loop)
                                    if len(register) == 0:                           # 第一次擷取，register是空的array
                                        register = result_npy
                                        cv2.putText(regist_img, str(cou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        cv2.imshow('register 0', regist_img)
                                        cou += 1
                                        cap_flag = 1
                                    elif cou < register_num:                         # press '1' will no function, if features are enough
                                        register = np.concatenate((register, result_npy), axis=0)
                                        cv2.putText(regist_img, str(cou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        cv2.imshow('register 0', regist_img)
                                        cou += 1
                                        cap_flag = 1
                                elif key == 27:
                                    for cam_id_ in range(len(caps)):
                                        caps[cam_id_].stop()
                                    cv2.destroyAllWindows()
                                    exit()


def main():
    
    video_root = 'D:/code/tf-openpose_new/one_million.mp4'
    #video_root = 'D:/code/tf-openpose_new/baseball_video/baseball_1.mp4'
    # save_root = 'D:/Code/MultiCamOverlap/dataset/detections/No3/'
    # MODEL_NAME
    MODEL_NAME = 'C:/Users/ray20/Desktop/Michael/Code/weight/faster_rcnn_inception_v2_coco_2018_01_28'
    #MODEL_NAME = 'C:/Users/ray20/Desktop/Michael/Code/weight/ssd_mobilenet_v2_coco_2018_03_29'
    # Path to frozen detection graph. This is the actual model
    # - that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    #PATH_TO_LABELS = os.path.join('data_object', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = 'C:/Users/ray20/Desktop/Michael/Code/models/research/object_detection/data/mscoco_label_map.pbtxt'
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

    print('Regist?(y/n)')
    ans = input()
    if ans == 'n' or ans == 'N':
        if os.path.isfile('feature.npy'):
            register = np.load('feature.npy')
            object_detection(detection_graph, video_root, category_index, register)
        else:
            print('Cannot find feature file, please regist first')
            exit()
    elif ans == 'y' or ans == 'Y':
        print('Regis frame number : (multiples of 4)')
        reg_num = input()
        if is_num(reg_num):
            if int(reg_num) % 4 == 0:
                register = regist(detection_graph, category_index, int(reg_num))
                object_detection(detection_graph, video_root, category_index, register)
            else:
                print('please input the multiples of 4')
                exit()
        else:
            print('please input integer')
            exit()
    else:
        print('please input y or n')
        exit()


if __name__ == '__main__':
    main()
