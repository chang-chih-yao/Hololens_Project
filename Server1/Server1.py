import numpy as np
import socket
import time
import cv2
import threading
import argparse
import logging
import datetime

import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from tsn_pytorch.models import TSN
from tsn_pytorch.transforms import *

import tkinter

'''
action_label -> action_name
1  -> 1         (No Action)
2  -> 2_start   (螺旋丸)
3  -> 2_end
4  -> 3_start   (甩)
5  -> 3_end
6  -> 4_start   (龜派氣功)
7  -> 4_end
8  -> 5_start   (落雷)
9  -> 5_end
10 -> 6         (防禦壹之型)
11 -> 7         (防禦貳之型) 太極
12 -> 8         (防禦參之型) 黑豹
13 -> 9         (防禦肆之型) 體操?
14 -> 10        (防禦伍之型) 結印
15 -> 11        (踢)
'''

################################### OpenPose ###########################################
gpu_options = tf.GPUOptions(allow_growth=True)
#e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
#e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))

############################## Action Recognition ######################################
global num_class
num_class = 15

model = TSN(num_class, 3, 'RGB',
            base_model='resnet34',
            consensus_type='avg', dropout=0.7)

checkpoint = torch.load('D:\\Code\\Hololens_Project\\Core\\tsn_pytorch\\pth\\holo_2019_1023_11_actions_15_class_MOD_4.pth')
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
HOST = '192.168.60.2'
PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # tcp
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # reuse tcp
sock.bind((HOST, PORT))
sock.listen(5)
print('Wait for connection...')


global_action_p0 = 0  # outside camera action
global_action_p1 = 0
gamepoint_p0 = 10
gamepoint_p1 = 10
p0_win_lose = 0       # 1->win, 2->lose
p1_win_lose = 0       # 1->win, 2->lose
holo_action_p0 = 0
holo_action_p1 = 0
defense_skill_2_p0 = 0  # p0 針對 Skill2 有沒有防禦成功
defense_skill_2_p1 = 0
blood_effect_p0 = 0
blood_effect_p1 = 0

status_data_p0 = b'0,0,0|'
status_data_p1 = b'0,0,0|'
data_cp = b''
frame_size_cp = None
co_str_cp = ''

temp_global_cou_p0 = 0
temp_global_cou_p1 = 0




def validate(model, rst):

    input_var = torch.autograd.Variable(rst, volatile=True)
    output = model(input_var)

    topk = (1,5)
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    top1 = int(pred[0][0])
    top2 = int(pred[1][0])
    top3 = int(pred[2][0])
    top4 = int(pred[3][0])
    top5 = int(pred[4][0])
    
    return output, pred



class TServer(threading.Thread):
    def __init__(self, socket, adr, lock, count=1, action=0, fps_time=0):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address= adr
        self.count = count
        self.action = action
        self.fps_time = fps_time
        self.lock = lock

        ##################### Kalman filter ##################
        '''
        它有3个输入参数
        dynam_params  ：状态空间的维数，这里为4
        measure_param ：测量值的维数，这里也为2
        control_params：控制向量的维数，默认为0。由于这里该模型中并没有控制变量，因此也为0。
        kalman.processNoiseCov    ：为模型系统的噪声，噪声越大，预测结果越不稳定，越容易接近模型系统预测值，且单步变化越大，相反，若噪声小，则预测结果与上个计算结果相差不大。
        kalman.measurementNoiseCov：为测量系统的协方差矩阵，方差越小，预测结果越接近测量值
        '''
        self.kalman_RElbow = cv2.KalmanFilter(4,2)
        self.kalman_RElbow.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman_RElbow.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman_RElbow.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1e-4
        self.kalman_RElbow.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.04
        self.kalman_RElbow.errorCovPost = np.array([[1,0],[0,1]], np.float32) * 1

        self.kalman_RWrist = cv2.KalmanFilter(4,2)
        self.kalman_RWrist.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman_RWrist.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman_RWrist.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1e-4
        self.kalman_RWrist.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.04
        self.kalman_RWrist.errorCovPost = np.array([[1,0],[0,1]], np.float32) * 1

        self.ori_RElbow = np.array([[0],[0]],np.float32)
        self.pre_RElbow = np.array([[0],[0]],np.float32)
        self.ori_RWrist = np.array([[0],[0]],np.float32)
        self.pre_RWrist = np.array([[0],[0]],np.float32)

    def run(self):
        print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
        print ('Client %s:%s connected.' % self.address)
        window_name = str(self.address[1])
        #print('port : %s' % window_name)

        ID = self.socket.recv(1024)
        print(ID)
        self.socket.send(b'OK')

        if ID == b'holo_P0':
            self.run_holo(window_name, 'P0')
        elif ID == b'holo_P1':
            self.run_holo(window_name, 'P1')
        elif ID == b'4cam':
            self.run_4cam(window_name)
        elif ID == b'Unity_demo':
            self.run_demo(window_name)
        elif ID == b'bye':
            print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
            print ('Client %s:%s disconnected.' % self.address)
            self.socket.close()
        else:
            print('wrong ID' + ' : ' + str(ID))
            self.socket.close()

    def openpose_coordinate_to_str(self, key_points):
        my_str = ''
        for i in range(len(key_points)):  # 18個點，36個值(x,y)
            my_str = my_str + str(key_points[i][0]) + ',' + str(key_points[i][1]) + ','
        return my_str

    def run_holo_reset(self):
        global p0_win_lose
        global p1_win_lose
        global gamepoint_p0
        global gamepoint_p1
        p0_win_lose = 0
        p1_win_lose = 0
        gamepoint_p0 = 10
        gamepoint_p1 = 10

    def run_demo(self, window_name):
        global frame_size_cp
        global data_cp
        global co_str_cp


        while(True):
            zz = self.socket.recv(1024)
            if zz == b'bye':
                break
            # print(zz)
            # frame_size_int = int.from_bytes(frame_size, byteorder='big')
            # print(frame_size_int)
            self.socket.send(frame_size_cp)
            self.socket.send(data_cp)
            self.socket.send(co_str_cp)

        print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
        print ('Client %s:%s disconnected. (run_demo)' % self.address)
        self.socket.close()


    def run_holo(self, window_name, player):

        global global_action_p0, global_action_p1
        global gamepoint_p0, gamepoint_p1
        global p0_win_lose, p1_win_lose
        global holo_action_p0, holo_action_p1
        global defense_skill_2_p0, defense_skill_2_p1
        global status_data_p0, status_data_p1
        global blood_effect_p0, blood_effect_p1

        global temp_global_cou_p0, temp_global_cou_p1

        global frame_size_cp
        global data_cp
        global co_str_cp

        images = list()

        while(True):

            no_human = 0   # no human flag
            data = b''
            try:
                temp_data = self.socket.recv(4096)
                frame_size = temp_data[:4]
                frame_size_int = int.from_bytes(frame_size, byteorder='big')
                #print(player + ' ' + str(frame_size_int))
                temp_data = temp_data[4:]

                data += temp_data
                while(True):
                    if len(data) == frame_size_int + 256:
                        break
                    temp_data = self.socket.recv(4096)
                    data += temp_data
            except ConnectionResetError:    # 當 hololens 關閉時
                self.run_holo_reset()
                break
            except ConnectionAbortedError:  # 當 hololens 關閉時
                self.run_holo_reset()
                break

            img_data = data[0:frame_size_int]              # 封包的前段是 HoloLens 傳過來的 image data
            frame = np.fromstring(img_data, dtype=np.uint8)
            dec_img = cv2.imdecode(frame, 1)
            crop_img = dec_img.copy()

            if player == 'P0':
                status_data_p0 = data[frame_size_int:]     # 封包的後段才是 HoloLens 傳過來的 status, status = "被對方的技能(Skill_2)打到, end game, start new game"
            elif player == 'P1':
                status_data_p1 = data[frame_size_int:]

            # if status_data_p0[0] == b'1'[0] or status_data_p0[2] == b'1'[0] or status_data_p0[4] == b'1'[0]:
            #     print(status_data_p0.split(b'|')[0])
            # else:
            #     print('------------------')

            humans = e.inference(dec_img, resize_to_default=True, upsample_size=4.0)
            img, key_points = TfPoseEstimator.draw_one_human(dec_img, humans, imgcopy=False, score=0.8)

            ####################### Kalman filter #######################
            if(key_points[3][0] != 0 or key_points[3][1] != 0):
                self.ori_RElbow = np.array([[key_points[3][0]],[key_points[3][1]]], np.float32)
                self.kalman_RElbow.correct(self.ori_RElbow)
                self.pre_RElbow = self.kalman_RElbow.predict()
            
            if(key_points[4][0] != 0 or key_points[4][1] != 0):
                self.ori_RWrist = np.array([[key_points[4][0]],[key_points[4][1]]], np.float32)
                self.kalman_RWrist.correct(self.ori_RWrist)
                self.pre_RWrist = self.kalman_RWrist.predict()

            key_points[3][0] = self.pre_RElbow[0,0]
            key_points[3][1] = self.pre_RElbow[1,0]
            key_points[4][0] = self.pre_RWrist[0,0]
            key_points[4][1] = self.pre_RWrist[1,0]


            co_str = self.openpose_coordinate_to_str(key_points)

            cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - self.fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(window_name + player, img)


            if key_points[1][0] == 0 or key_points[1][1] == 0:    # no human
                x1 = 224 - 126
                x2 = 224 + 126
                no_human = 1
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
                rst = trans(images)
                rst = torch.unsqueeze(rst, 0)
                my_output, my_pred = validate(model, rst)         # predict action

                self.action = int(my_pred[0][0])
                top2 = int(my_pred[1][0])
                top3 = int(my_pred[2][0])
                top4 = int(my_pred[3][0])
                top5 = int(my_pred[4][0])

                if no_human == 0:
                    detail = '{:d} ({:.2f}), {:d} ({:.2f}), {:d} ({:.2f}), {:d} ({:.2f}), {:d} ({:.2f})'.format(
                        self.action, float(my_output[0][int(my_pred[0][0])]),
                        top2, float(my_output[0][top2]),
                        top3, float(my_output[0][top3]),
                        top4, float(my_output[0][top4]),
                        top5, float(my_output[0][top5])
                    )
                    #print(self.action)
                    print(detail)

                if (self.action == 0):
                    self.action = num_class

                del images[0]

                if no_human == 1:      # if no human
                    holo_action_p0 = 1
                    holo_action_p1 = 1
                    # if player == 'P0':
                    #     if global_action_p1 == 2:
                    #         temp_global_cou_p1 += 1
                    #     else:
                    #         temp_global_cou_p1 = 0
                    #     if temp_global_cou_p1 > 10:
                    #         global_action_p1 = 3
                    #     holo_action_p1 = global_action_p1
                    # elif player == 'P1':
                    #     if global_action_p0 == 2:
                    #         temp_global_cou_p0 += 1
                    #     else:
                    #         temp_global_cou_p0 = 0
                    #     if temp_global_cou_p0 > 10:
                    #         global_action_p0 = 3
                    #     holo_action_p0 = global_action_p0
                else:
                    if player == 'P0':
                        holo_action_p1 = self.action
                    elif player == 'P1':
                        holo_action_p0 = self.action
                        
            elif self.count % 3 == 0:
                images.extend([img_tsn])
                

            if player == 'P0':
                if status_data_p0[2] == b'1'[0]:     # HoloLens那端已經進入end game畫面
                    holo_action_p1 = 1               # 遊戲 reset 階段辨識到的動作一率為 no action(1)
            elif player == 'P1':
                if status_data_p1[2] == b'1'[0]:
                    holo_action_p0 = 1

            co_str = co_str + str(self.count) + ',' + str(holo_action_p0) + ',' + str(holo_action_p1)
            co_str = co_str + ',' + str(gamepoint_p0) + ',' + str(gamepoint_p1) + ',' + str(p0_win_lose) + ',' + str(p1_win_lose)
            co_str = co_str + ',' + str(defense_skill_2_p0) + ',' + str(defense_skill_2_p1) + ',' + str(blood_effect_p0) + ',' + str(blood_effect_p1)
            #print(co_str)
            co_str = bytes(co_str, 'ascii')

            ####################### for unity_demo img_data #####################
            if player == 'P0':
                frame_size_cp = frame_size
                data_cp = img_data
                co_str_cp = co_str
            
            try:
                self.socket.send(co_str)
            except ConnectionResetError:    # 當 hololens 關閉時
                self.run_holo_reset()
                break
            except ConnectionAbortedError:  # 當 hololens 關閉時
                self.run_holo_reset()
                break

            self.fps_time = time.time()
            self.count += 1

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyWindow(window_name + player)
        print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
        print ('Client %s:%s disconnected. (HoloLens)' % self.address)
        self.socket.close()

    def run_4cam(self, window_name):
        global global_action_p0
        global global_action_p1
        while(True):
            action_byte = self.socket.recv(1024)
            if action_byte == b'bye':
                break
            #print(action_byte)
            act_p0 = int(bytes.decode(action_byte).split(',')[0])
            act_p1 = int(bytes.decode(action_byte).split(',')[1])

            global_action_p0 = int((act_p0 + 2)/4) + 1
            global_action_p1 = int((act_p1 + 2)/4) + 1

            print(str(global_action_p0) + ', ' + str(global_action_p1))

            '''
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
            '''
                
            self.socket.send(b'OK')

        print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
        print ('Client %s:%s disconnected. (run_4cam)' % self.address)
        self.socket.close()

class GameSystem(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global gamepoint_p0, gamepoint_p1
        global p0_win_lose, p1_win_lose
        global holo_action_p0, holo_action_p1
        global global_action_p0, global_action_p1
        global defense_skill_2_p0, defense_skill_2_p1
        global blood_effect_p0, blood_effect_p1

        skill_2_damage = 5
        #action2_start_p0 = 0
        #action2_start_p1 = 0

        skill_wait_time = 1      # 對方施放每一招，都會有一個等待時間(硬直2秒)
        restart_wait_time = 5    # GameSystem 重啟等待時間

        while(True):
            ############################# p0 看 p1 #############################
            if status_data_p0[0] == b'1'[0]:        # 如果p0被技能(Skill_2)打到
                if holo_action_p0 != 10:             # 如果p0 沒有 做防禦動作
                    blood_effect_p0 = 1             # p0 受傷噴血的動畫，這個會透過socket傳給hololens
                    gamepoint_p0 -= skill_2_damage  # 扣血
                else:                               # 如果p0 有 做防禦動作
                    defense_skill_2_p0 = 1          # 成功防禦，這個會透過socket傳給hololens，顯示防禦特效
                
                if gamepoint_p0 != 0:
                    time.sleep(0.5)
                    blood_effect_p0 = 0             # init
                    defense_skill_2_p0 = 0          # init
                    time.sleep(skill_wait_time)
            '''
            if holo_action_p1 == 2:
                action2_start_p1 = 1
            elif holo_action_p1 == 3 and action2_start_p1 == 1:
                f = 0
                for i in range(3):
                    time.sleep(0.1)
                    if holo_action_p1 != 3:
                        f = 1
                        break
                if f == 0:                      # 代表連續0.6秒(0.1*6)，都是這個動作，判定對方確實是在做這個動作
                    if holo_action_p0 != 7:     # 如果另一方 沒有 做防禦動作
                        gamepoint_p0 -= skill_2_damage
                    else:                       # 如果另一方 有 做防禦動作
                        defense_skill_2_p0 = 1  # 成功防禦
                    if gamepoint_p0 != 0:
                        time.sleep(skill_wait_time)
                        defense_skill_2_p0 = 0  # init
            else:
                action2_start_p1 = 0
            '''

            ############################# p1 看 p0 #############################
            if status_data_p1[0] == b'1'[0]:  # 如果p1被技能(Skill_2)打到
                if holo_action_p1 != 10:       # 如果p1 沒有 做防禦動作
                    blood_effect_p1 = 1
                    gamepoint_p1 -= skill_2_damage
                else:                         # 如果p1 有 做防禦動作
                    defense_skill_2_p1 = 1
                
                if gamepoint_p1 != 0:
                    time.sleep(0.5)
                    blood_effect_p1 = 0       # init
                    defense_skill_2_p1 = 0    # init
                    time.sleep(skill_wait_time)
            
            ############################# 遊戲結束，結算，reset #############################
            if gamepoint_p0 == 0:
                print('p0 lose, p1 win')
                p0_win_lose = 2     # p0 lose
                p1_win_lose = 1     # p1 win
                time.sleep(0.5)
                blood_effect_p0 = 0 # init
                # ------------ reset data ------------ #
                p0_win_lose = 0
                p1_win_lose = 0
                gamepoint_p0 = 10
                gamepoint_p1 = 10
                holo_action_p0 = 1
                holo_action_p1 = 1
                global_action_p0 = 1
                global_action_p1 = 1
                print('---------- Game system ready ----------')
                time.sleep(1)
                
            elif gamepoint_p1 == 0:
                print('p0 win, p1 lose')
                p0_win_lose = 1     # p0 lose
                p1_win_lose = 2     # p1 win
                time.sleep(0.5)
                blood_effect_p1 = 0 # init
                # ------------ reset data ------------ #
                p0_win_lose = 0
                p1_win_lose = 0
                gamepoint_p0 = 10
                gamepoint_p1 = 10
                holo_action_p0 = 1
                holo_action_p1 = 1
                global_action_p0 = 1
                global_action_p1 = 1
                print('---------- Game system ready ----------')
                time.sleep(1)
                
            time.sleep(0.03)
        

if __name__ == '__main__':
    lock = threading.Lock()
    GameSystem().start()

    while True:
        (client, adr) = sock.accept()
        TServer(client, adr, lock, count=1, action=0, fps_time=0).start()


