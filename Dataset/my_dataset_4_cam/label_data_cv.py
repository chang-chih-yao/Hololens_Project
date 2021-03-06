import os
from shutil import copyfile
import cv2

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    #print len(channels)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

root_folder = 'D:/Code/Hololens_Project/Dataset/my_dataset_4_cam/'

while(True):
    print('Which person dataset : ', end='')
    person = input()
    if person == 'Q' or person == 'q' or person == 'exit':
        break
    
    while(True):
        print('Action label         : ', end='')
        action_label = input()
        if action_label == 'Q' or action_label == 'q' or action_label == 'exit':
            break
        
        action_str = ''
        if action_label == '1':
            action_str = 'BG'
        elif action_label == '2':
            action_str = '螺旋丸'
        elif action_label == '3':
            action_str = '甩鞭'
        elif action_label == '4':
            action_str = '龜派氣功'
        elif action_label == '5':
            action_str = '大絕'
        elif action_label == '6':
            action_str = '防禦'
        else:
            print('type error')
            continue

        folder_num = len(os.listdir(root_folder + action_label + '/' + '0/'))
        begin_folder = folder_num + 1
        print('Begin folder         :', begin_folder)

        print('Person: \'' + person + '\', Action: \'' + action_str + '\', Begin folder: \'' + str(begin_folder) + '\'')
        print('----------------------------------------------------------')

        print('Start image index:')
        start_img = input()

        img_cou = int(start_img)
        start_index = 0
        end_index = 0
        record_end_index = 0

        while(True):
            img_path = root_folder + 'Raw_data/' + person + '/0/img_{:05d}.jpg'.format(img_cou)
            if not os.path.exists(img_path):
                print('end')
                cv2.destroyAllWindows()
                break
            img = cv2.imread(img_path)
            if start_index != 0:
                cv2.putText(img, "Start", (800, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            #img = hisEqulColor(img)
            cv2.imshow('label_data', img)
            key = cv2.waitKey(20)
            if key == 27:
                f = open('end_index.txt', 'w')
                f.writelines(str(record_end_index) + '\n')
                f.writelines(str(img_cou))
                f.close()
                print(str(record_end_index), str(img_cou))
                cv2.destroyAllWindows()
                break
            elif key == 104:               # h
                img_cou += 1
            elif key == 102:               # f
                img_cou -= 1
            elif key == 106:               # j
                img_cou += 5
            elif key == 100:                # d
                img_cou -= 5
            elif key == 32:                # space
                if start_index == 0:
                    start_index = img_cou
                else:
                    if start_index != img_cou:
                        end_index = img_cou
                        record_end_index = end_index
                        
                        folder_name = '{:04d}'.format(begin_folder)
                        for direction in range(4):
                            if not os.path.exists(root_folder + action_label + '/' + str(direction) + '/' + folder_name + '/'):
                                os.mkdir(root_folder + action_label + '/' + str(direction) + '/' + folder_name + '/')
                                
                                for i in range(start_index, end_index + 1):
                                    copyfile(root_folder + 'Raw_data/' + person + '/' + str(direction) + '/img_{:05d}.jpg'.format(i), 
                                            root_folder + action_label + '/' + str(direction) + '/' + folder_name + '/img_{:05d}.jpg'.format(i - start_index + 1))
                            else:
                                print('Folder existed !!!!!!!!!!!!!!!!!!!!!!')
                                break
                        print(folder_name ,'has been created. Copy from', str(start_index), 'to', str(end_index), '({:d})'.format( end_index - start_index + 1 ))

                        begin_folder += 1
                        start_index = 0
                        end_index = 0
            elif key == 8:                 # backspace
                if start_index != 0:
                    start_index = 0
            
