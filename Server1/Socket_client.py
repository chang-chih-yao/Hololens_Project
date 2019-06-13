import numpy as np
import socket
import time
import cv2
import threading
import datetime

player = input()
if player == '0':
    player = 'holo_P0'
elif player == '1':
    player = 'holo_P1'
else:
    print('error')
    exit()

################################### Socket #######################################
HOST = '192.168.11.107'
PORT = 9000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

s.send(bytes(player, 'ascii'))
data = s.recv(1024)
print(data)

while(True):
    input_data = input()
    s.send(bytes(input_data, 'ascii'))
    if input_data == 'bye':
        break
    data = s.recv(1024)
    print(data)