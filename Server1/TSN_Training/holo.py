import time
import cv2
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from models import TSN
from transforms import *

num_class = 0

def main():
    global num_class
    num_class = 2

    model = TSN(num_class, 3, 'RGB',
                base_model='resnet34',
                consensus_type='avg', dropout=0.7)
    
    checkpoint = torch.load('D:\\code\\Action Recognition\\tsn-pytorch\\pth\\my_rgb_checkpoint_0124.pth')
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

    trans = trans = torchvision.transforms.Compose([
        GroupScale(int(crop_size)),
        Stack(roll=False),
        ToTorchFormatTensor(div=True)
        ]
    )



    img_cv1 = cv2.imread('D:\\Dataset\\Action\\my_dataset\\crop\\1\\1_0001\\img_00001.jpg')
    img1 = Image.fromarray(cv2.cvtColor(img_cv1, cv2.COLOR_BGR2RGB))
    img_cv2 = cv2.imread('D:\\Dataset\\Action\\my_dataset\\crop\\1\\1_0001\\img_00002.jpg')
    img2 = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    img_cv3 = cv2.imread('D:\\Dataset\\Action\\my_dataset\\crop\\1\\1_0001\\img_00003.jpg')
    img3 = Image.fromarray(cv2.cvtColor(img_cv3, cv2.COLOR_BGR2RGB))

    images = list()
    images.extend([img1])
    images.extend([img2])
    images.extend([img3])

    rst = trans(images)
    rst = torch.unsqueeze(rst, 0)

    action = validate(model, rst)
    print(action)


def validate(model, rst):

    # switch to evaluate mode
    model.eval()

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