import time
import socket
import cv2
import numpy as np
import os
import threading

################################### Socket #######################################
HOST = '192.168.11.107'
PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # tcp
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # reuse tcp
sock.bind((HOST, PORT))
sock.listen(1)
print('Wait for connection...')

class TServer(threading.Thread):
    def __init__(self, socket, adr, count=1, fps_time=0):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address= adr
        self.count = count
        self.fps_time = fps_time
        

    def run(self):
        print('folder name:')
        my_str = input()
        if not os.path.exists('Raw_data/' + my_str + '/'):
            os.mkdir('Raw_data/' + my_str + '/')
        else:
            print('folder existed, continue?(y/n)')
            zz = input()
            if zz == 'n':
                exit()
        

        ID = self.socket.recv(1024)
        print(ID)
        self.socket.send(b'OK')

        while(True):

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
                break
            except ConnectionAbortedError:  # 當 hololens 關閉時
                break

            img_data = data[0:frame_size_int]
            # if player == 'P0':
            #     status_data_p0 = data[frame_size_int:]
            # elif player == 'P1':
            #     status_data_p1 = data[frame_size_int:]
            # if status_data_p0[0] == b'1'[0] or status_data_p0[2] == b'1'[0] or status_data_p0[4] == b'1'[0]:
            #     print(status_data_p0.split(b'|')[0])
            # else:
            #     print('------------------')

            frame = np.fromstring(img_data, dtype=np.uint8)
            dec_img = cv2.imdecode(frame, 1)


            img_dir = 'Raw_data/' + my_str + '/img_{:05d}.jpg'.format(self.count)
            cv2.imwrite(img_dir, dec_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            str_send = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,' + str(self.count) + ',0,0,10,10,0,0,0,0,0,0'
            str_send = bytes(str_send, 'ascii')
            self.socket.send(str_send)

            cv2.putText(dec_img,
                        "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', dec_img)
            self.fps_time = time.time()
            
            self.count += 1

            if cv2.waitKey(1) == 27:
                break
        
        cv2.destroyAllWindows()
        print('finished')

if __name__ == '__main__':
    while True:
        (client, adr) = sock.accept()
        TServer(client, adr, count=1, fps_time=0).start()