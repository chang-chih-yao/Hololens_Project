import argparse
import logging
import time
import socket

import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from models import TSN
from transforms import *

class tmp:
    def __init__(self, npimg, 
                LWrist_x=0, LWrist_y=0, LElbow_x=0, LElbow_y=0, LShoulder_x=0, LShoulder_y=0, 
                RWrist_x=0, RWrist_y=0, RElbow_x=0, RElbow_y=0, RShoulder_x=0, RShoulder_y=0,
                Neck_x=0, Neck_y=0):
        self.npimg = npimg
        self.LWrist_x = LWrist_x
        self.LWrist_y = LWrist_y
        self.LElbow_x = LElbow_x
        self.LElbow_y = LElbow_y
        self.LShoulder_x = LShoulder_x
        self.LShoulder_y = LShoulder_y

        self.RWrist_x = RWrist_x
        self.RWrist_y = RWrist_y
        self.RElbow_x = RElbow_x
        self.RElbow_y = RElbow_y
        self.RShoulder_x = RShoulder_x
        self.RShoulder_y = RShoulder_y

        self.Neck_x = Neck_x
        self.Neck_y = Neck_y


num_class = 0

def main():
    HOST = '192.168.1.106'  # Standard loopback interface address (localhost)
    PORT = 9000
    fps_time = 0

    ################################### OpenPose ###########################################
    w, h = model_wh('432x368')
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
    #e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))

    
    ############################## Action Recognition ######################################
    global num_class
    num_class = 2

    model = TSN(num_class, 3, 'RGB',
                base_model='resnet34',
                consensus_type='avg', dropout=0.7)
    
    checkpoint = torch.load('D:\\code\\Action Recognition\\tsn-pytorch\\pth\\my_rgb_checkpoint_0127_2.pth')
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()

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

    images = list()


    ################################### Socket #######################################
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    #s.bind((HOST, PORT))
    #s.listen(1)
    #print('Waiting for connection...')
    #sock, addr = s.accept()
    #print('Accept new connection from %s:%s...' % addr)
    print("Connect")

    count = 1
    action = 0

    while(True):
        
        data = b''

        '''
        # bufsize = 4096 * 27                     # 4096 * 27 = 110592
        bufsize = 112896
        while(bufsize):
            temp_data = s.recv(4096)
            bufsize -= len(temp_data)
            data += temp_data

        # now data.length = 335872
        # the image is 252*448, and 3 channel -> image.length = 252 * 448 * 3 = 338688
        # so 338688 - 335872 = 2816 
        
        # data += b'0' * 2304
        image_byte = np.fromstring(data, dtype='uint8')

        image_byte = np.reshape(image_byte, (252, 448))
        image = np.zeros([252, 448, 3], dtype='uint8')
        image[:, :, 0] = (image_byte[:, :] / 64).astype(np.uint8) * 64 + 32                 # Blue
        image[:, :, 1] = (image_byte[:, :] / 8).astype(np.uint8) % 8 * 32 + 16              # Green
        image[:, :, 2] = (image_byte[:, :] % 8).astype(np.uint8) * 32 + 16                  # Red   

        #print('len(image_byte):', len(image_byte))
        '''

        temp_data = s.recv(4096)
        frame_size = temp_data[:4]
        frame_size_int = int.from_bytes(frame_size, byteorder='big')
        #print(frame_size_int)
        temp_data = temp_data[4:]
        data += temp_data
        while(True):
            if len(data) == frame_size_int:
                break
            temp_data = s.recv(4096)
            data += temp_data
            
        frame = np.fromstring(data, dtype=np.uint8)
        dec_img = cv2.imdecode(frame, 1)
        
    
        crop_img = dec_img.copy()


        humans = e.inference(dec_img)
        temp_class = TfPoseEstimator.draw_humans(dec_img, humans, imgcopy=False)

        co_str = str(temp_class.LWrist_x) + ',' + str(temp_class.LWrist_y) + ',' + str(temp_class.LElbow_x) + ',' + str(temp_class.LElbow_y) + ',' + str(temp_class.LShoulder_x) + ',' + str(temp_class.LShoulder_y)
        co_str = co_str + ',' + str(temp_class.RWrist_x) + ',' + str(temp_class.RWrist_y) + ',' + str(temp_class.RElbow_x) + ',' + str(temp_class.RElbow_y) + ',' + str(temp_class.RShoulder_x) + ',' + str(temp_class.RShoulder_y)
        co_str = co_str + ',' + str(temp_class.Neck_x) + ',' + str(temp_class.Neck_y)

        cv2.putText(temp_class.npimg,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation', temp_class.npimg)


        if temp_class.Neck_x == 0 or temp_class.Neck_y == 0:
            x1 = 224 - 126
            x2 = 224 + 126
        elif temp_class.Neck_x-126 < 0:
            x1 = 0
            x2 = 252
        elif temp_class.Neck_x+126 > 447:
            x1 = 196
            x2 = 448
        else:
            x1 = temp_class.Neck_x - 126
            x2 = temp_class.Neck_x + 126

        crop_img = crop_img[:, x1:x2, :]
        #cv2.imshow('TSN', crop_img)

        img_tsn = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

        if count%6 == 0:
            images.extend([img_tsn])
            # print(images[0].size)
            # print(images[1].size)
            # print(images[2].size)
            rst = trans(images)
            rst = torch.unsqueeze(rst, 0)
            action = validate(model, rst)
            
            images.clear()
        elif count%2 == 0:
            images.extend([img_tsn])
            


        co_str = co_str + ',' + str(count) + ',' + str(action)
        co_str = bytes(co_str, 'ascii')
        print(co_str)
        s.send(co_str)
        
        fps_time = time.time()
        

        count += 1

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    print('finished')


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


if __name__ == '__main__':
    main()