import numpy as np
import socket
import time
import cv2
import threading
import datetime

################################### Socket #######################################
HOST = '192.168.11.107'
PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # tcp
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # reuse tcp
sock.bind((HOST, PORT))
sock.listen(4)
print('Wait for connection...')

class TServer(threading.Thread):
    def __init__(self, socket, adr, count=1, action=0, fps_time=0):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address= adr
        self.count = count
        self.action = action
        self.fps_time = fps_time
        #self.lock = lock
        

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
        elif ID == b'bye':
            print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
            print ('Client %s:%s disconnected.' % self.address)
            self.socket.close()
        else:
            print('wrong ID' + ' : ' + str(ID))
            self.socket.close()

    def run_holo(self, window_name, player):
        while(True):
            data = self.socket.recv(10)
            print(data, str(len(data)))
            if data == b'bye':
                break
            status = 'you are ' + player
            self.socket.send(bytes(status, 'ascii'))

        print(datetime.datetime.now().strftime('%m/%d %H:%M:%S '), end='')
        print ('Client %s:%s disconnected. (HoloLens)' % self.address)
        self.socket.close()


if __name__ == '__main__':
    #lock = threading.Lock()
    #GameSystem().start()

    while True:
        (client, adr) = sock.accept()
        TServer(client, adr, count=1, action=0, fps_time=0).start()