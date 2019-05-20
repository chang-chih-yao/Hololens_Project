import numpy as np
import socket
import time
import cv2
import threading
import argparse
import logging

import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from tsn_pytorch.models import TSN
from tsn_pytorch.transforms import *

################################### OpenPose ###########################################
gpu_options = tf.GPUOptions(allow_growth=True)
#e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
#e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))

############################## Action Recognition ######################################
global num_class
num_class = 6

model = TSN(num_class, 3, 'RGB',
            base_model='resnet34',
            consensus_type='avg', dropout=0.7)

checkpoint = torch.load('D:\\Code\\Action_Recognition\\tsn_pytorch\\pth\\holo_2019_0314_6_actions.pth')
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



################################### Socket #######################################
HOST = '192.168.208.231'
PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # tcp
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # reuse tcp
sock.bind((HOST, PORT))
sock.listen(3)
print('Wait for connection...')

global_action = 0

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

def openpose_coordinate_to_str(key_points):
    my_str = ''
    for i in range(len(key_points)):
        my_str = my_str + str(key_points[i][0]) + ',' + str(key_points[i][1]) + ','
    return my_str

class TServer(threading.Thread):
    def __init__(self, socket, adr, count, action, fps_time):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address= adr
        self.count = count
        self.action = action
        self.fps_time = fps_time
        

    def run(self):
        print ('Client %s:%s connected.' % self.address)
        window_name = str(self.address[1])
        print('port : %s' % window_name)

        ID = self.socket.recv(1024)
        print(ID)
        self.socket.send(b'OK')

        if ID == b'holo':
            self.run_holo(window_name)
        elif ID == b'4cam':
            self.run_4cam(window_name)
        else:
            print('wrong ID' + ' : ' + str(ID))
            self.socket.close()

    def run_holo(self, window_name):

        global global_action

        images = list()

        while(True):
        
            data = b''

            temp_data = self.socket.recv(4096)
            frame_size = temp_data[:4]
            frame_size_int = int.from_bytes(frame_size, byteorder='big')
            #print(frame_size_int)
            temp_data = temp_data[4:]
            data += temp_data
            while(True):
                if len(data) == frame_size_int:
                    break
                temp_data = self.socket.recv(4096)
                data += temp_data
                
            frame = np.fromstring(data, dtype=np.uint8)
            dec_img = cv2.imdecode(frame, 1)
            
        
            crop_img = dec_img.copy()


            humans = e.inference(dec_img, resize_to_default=True, upsample_size=4.0)
            key_points = TfPoseEstimator.get_keypoints(dec_img, humans)
            img = TfPoseEstimator.draw_one_human(dec_img, humans, imgcopy=False, score=0.8)

            co_str = openpose_coordinate_to_str(key_points)

            cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(window_name, img)


            if key_points[1][0] == 0 or key_points[1][1] == 0:
                x1 = 224 - 126
                x2 = 224 + 126
            elif key_points[1][0]-126 < 0:
                x1 = 0
                x2 = 252
            elif key_points[1][0]+126 > 447:
                x1 = 196
                x2 = 448
            else:
                x1 = key_points[1][0] - 126
                x2 = key_points[1][0] + 126

            crop_img = crop_img[:, x1:x2, :].copy()
            #cv2.imshow('TSN', crop_img)

            img_tsn = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

            if self.count % 3 == 0 and len(images) == 2:
                images.extend([img_tsn])
                # print(images[0].size)
                # print(images[1].size)
                # print(images[2].size)
                rst = trans(images)
                rst = torch.unsqueeze(rst, 0)
                self.action = validate(model, rst)
                
                del images[0]
            elif self.count % 3 == 0:
                images.extend([img_tsn])
                


            co_str = co_str + str(self.count) + ',' + str(self.action) + ',' + str(global_action)
            co_str = bytes(co_str, 'ascii')
            #print(co_str, str(global_action))
            self.socket.send(co_str)
            
            self.fps_time = time.time()
            

            self.count += 1

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyWindow(window_name)
        print ('Client %s:%s disconnected.' % self.address)
        self.socket.close()

    def run_4cam(self, window_name):
        global global_action
        while(True):
            action_byte = self.socket.recv(1024)
            if action_byte == b'bye':
                break
            print(action_byte)
            if action_byte == b'1':
                global_action = 1
            elif action_byte == b'2' or action_byte == b'3' or action_byte == b'4' or action_byte == b'5':
                global_action = 2
            elif action_byte == b'6' or action_byte == b'7' or action_byte == b'8' or action_byte == b'9':
                global_action = 3
            elif action_byte == b'10' or action_byte == b'11' or action_byte == b'12' or action_byte == b'13':
                global_action = 4
            elif action_byte == b'14' or action_byte == b'15' or action_byte == b'16' or action_byte == b'17':
                global_action = 5
            elif action_byte == b'18' or action_byte == b'19' or action_byte == b'20' or action_byte == b'21':
                global_action = 6
                
            self.socket.send(b'OK')

        print ('Client %s:%s disconnected.' % self.address)
        self.socket.close()

if __name__ == '__main__':

    while True:
        (client, adr) = sock.accept()
        TServer(client, adr, 1, 0, 0).start()


