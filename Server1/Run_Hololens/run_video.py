import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    #logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(args.video)
    #ret_val, image = cap.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    if (cap.isOpened()== False):
        print("Error opening video stream or file")


    frame_array = []


    while(cap.isOpened()):
        ret_val, image = cap.read()
        if ret_val == False:
            break

        image_crop = image[::2, ::2, :].copy()

        humans = e.inference(image_crop)
        image_crop = TfPoseEstimator.draw_humans(image_crop, humans, imgcopy=False)

        #logger.debug('show+')

        frame_array.append(image_crop.npimg)
        
        cv2.putText(image_crop.npimg,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        
        
        height, width, layers = image_crop.npimg.shape
        size = (width,height)
        cv2.imshow('tf-pose-estimation result', image_crop.npimg)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    
    out = cv2.VideoWriter('old.mp4',cv2.VideoWriter_fourcc(*'MP42'), 25, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    

    cap.release()

    cv2.destroyAllWindows()
logger.debug('finished+')
