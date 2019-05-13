import time
import socket
import cv2
import numpy as np
import os

def main():
    print('folder name:')
    my_str = input()
    if not os.path.exists('D:/Dataset/Action/my_dataset/Raw_data/' + my_str + '/'):
        os.mkdir('D:/Dataset/Action/my_dataset/Raw_data/' + my_str + '/')
    else:
        print('folder existed, continue?(y/n)')
        zz = input()
        if zz == 'n':
            exit()

    HOST = '192.168.11.130'  # Standard loopback interface address (localhost)
    PORT = 9000

    fps_time = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print("Connect")

    count = 1
    data = b''

    while(True):
        data = b''

        temp_data = s.recv(4096)
        frame_size = temp_data[:4]
        frame_size_int = int.from_bytes(frame_size, byteorder='big')
        print(frame_size_int)
        temp_data = temp_data[4:]
        data += temp_data
        while(True):
            if len(data) == frame_size_int:
                break
            temp_data = s.recv(4096)
            data += temp_data


        frame = np.fromstring(data, dtype=np.uint8)
        dec_img = cv2.imdecode(frame, 1)


        img_dir = 'D:/Dataset/Action/my_dataset/Raw_data/' + my_str + '/img_{:05d}.jpg'.format(count)
        cv2.imwrite(img_dir, dec_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        str_send = '0,0,0,0,0,0,0,0,0,0,0,0,0,0' + ',' + str(count) + ',0'
        str_send = bytes(str_send, 'ascii')
        s.send(str_send)
        #s.send(b'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
        cv2.putText(dec_img,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', dec_img)
        fps_time = time.time()
        
        count += 1

        if cv2.waitKey(10) == 27:
            break
    
    cv2.destroyAllWindows()
    print('finished')

if __name__ == '__main__':
    main()