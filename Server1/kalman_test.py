import time
import cv2
import threading
import numpy as np

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
print('start OpenPose')
gpu_options = tf.GPUOptions(allow_growth=True)
#e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
e = TfPoseEstimator(get_graph_path('mobilenet_v2_large'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
#e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368), tf_config=tf.ConfigProto(gpu_options=gpu_options))
print('end OpenPose')
############################## Action Recognition ######################################
global num_class
num_class = 7

model = TSN(num_class, 3, 'RGB',
            base_model='resnet34',
            consensus_type='avg', dropout=0.7)

checkpoint = torch.load('D:\\Code\\Hololens_Project\\Core\\tsn_pytorch\\pth\\holo_2019_0521_6_actions_7_class.pth')
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
model.load_state_dict(base_dict)

crop_size = model.crop_size

model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True

# switch to evaluate mode
model.eval()

trans = torchvision.transforms.Compose([
    GroupScale(int(crop_size)),
    Stack(roll=False),
    ToTorchFormatTensor(div=True)
    ]
)

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

##################### Kalman filter ##################
'''
它有3个输入参数
dynam_params  ：状态空间的维数，这里为4
measure_param ：测量值的维数，这里也为2
control_params：控制向量的维数，默认为0。由于这里该模型中并没有控制变量，因此也为0。
kalman.processNoiseCov    ：为模型系统的噪声，噪声越大，预测结果越不稳定，越容易接近模型系统预测值，且单步变化越大，相反，若噪声小，则预测结果与上个计算结果相差不大。
kalman.measurementNoiseCov：为测量系统的协方差矩阵，方差越小，预测结果越接近测量值
'''
kalman_RElbow = cv2.KalmanFilter(4,2)
kalman_RElbow.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman_RElbow.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman_RElbow.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1e-4
kalman_RElbow.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.05
kalman_RElbow.errorCovPost = np.array([[1,0],[0,1]], np.float32) * 1

kalman_RWrist = cv2.KalmanFilter(4,2)
kalman_RWrist.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman_RWrist.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman_RWrist.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 1e-4
kalman_RWrist.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.05
kalman_RWrist.errorCovPost = np.array([[1,0],[0,1]], np.float32) * 1

ori_RElbow = np.array([[0],[0]],np.float32)
pre_RElbow = np.array([[0],[0]],np.float32)
ori_RWrist = np.array([[0],[0]],np.float32)
pre_RWrist = np.array([[0],[0]],np.float32)


class cam(threading.Thread):
    def __init__(self, lock, flag='camera', cam_id=0, count=1):
        threading.Thread.__init__(self)
        self.flag = flag
        self.cam_id = cam_id
        self.count = count
        self.lock = lock

    def run(self):
        if self.flag == 'camera':
            self.run_cam(self.cam_id)
        elif self.flag == 'video':
            self.run_video()

    def run_cam(self, cam_id):
        global ori_RWrist, pre_RWrist, ori_RElbow, pre_RElbow

        my_cam = cv2.VideoCapture(cam_id)
        fps_time = 0
        images = list()

        while(True):
            ret_val, image = my_cam.read()

            kalmain_img = image.copy()

            humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
            #key_points = TfPoseEstimator.get_keypoints(image, humans)
            img, key_points = TfPoseEstimator.draw_one_human(image, humans, imgcopy=False, score=0.8)
            print(key_points)

            if(key_points[3][0] != 0 or key_points[3][1] != 0):
                ori_RElbow = np.array([[key_points[3][0]],[key_points[3][1]]], np.float32)
                kalman_RElbow.correct(ori_RElbow)
                pre_RElbow = kalman_RElbow.predict()
            
            if(key_points[4][0] != 0 or key_points[4][1] != 0):
                ori_RWrist = np.array([[key_points[4][0]],[key_points[4][1]]], np.float32)
                kalman_RWrist.correct(ori_RWrist)
                pre_RWrist = kalman_RWrist.predict()

            print(pre_RElbow[0,0])
            print(pre_RElbow[1,0])
            
            cv2.circle(kalmain_img, (pre_RElbow[0,0], pre_RElbow[1,0]), 4, (255,0,255), 5)
            cv2.circle(kalmain_img, (pre_RWrist[0,0], pre_RWrist[1,0]), 4, (255,0,0), 5)
            cv2.line(kalmain_img, (pre_RElbow[0,0], pre_RElbow[1,0]), (pre_RWrist[0,0], pre_RWrist[1,0]), (0,0,255), 2)
            cv2.imshow('kalman', kalmain_img)

            cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('camera' + str(cam_id), img)

            '''
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            img_tsn = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if self.count % 3 == 0 and len(images) == 2:
                images.extend([img_tsn])
                # print(images[0].size)
                # print(images[1].size)
                # print(images[2].size)
                rst = trans(images)
                rst = torch.unsqueeze(rst, 0)
                self.action = validate(model, rst)

                #print(str(self.action) + ' ' + str(cam_id))
                        
                del images[0]
            elif self.count % 3 == 0:
                images.extend([img_tsn])
            '''
            
            
            fps_time = time.time()
            self.count += 1
            if cv2.waitKey(1) == 27:
                break
    
    def run_video(self):
        cap = cv2.VideoCapture('Teach_action_new.mp4')
        fps_time = 0
        if cap.isOpened() is False:
            print("Error opening video stream or file")
        while cap.isOpened():
            ret_val, image = cap.read()

            humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
            key_points = TfPoseEstimator.get_keypoints(image, humans)
            img = TfPoseEstimator.draw_one_human(image, humans, imgcopy=False, score=0.8)

            cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('video', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break


if __name__ == '__main__':
    lock = threading.Lock()
    cam(lock, flag='camera', cam_id=0).start()
    #cam(lock, flag='camera', cam_id=1).start()
    #cam(flag='video').start()