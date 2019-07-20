import os
from shutil import copyfile

root_path = 'D:/Code/Hololens_Project/Dataset/my_dataset_holo/'

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

        folder_num = len(os.listdir(root_path + action_label + '/'))
        begin_folder = folder_num + 1
        print('Begin folder         :', begin_folder)

        print('Person: \'' + person + '\', Action: \'' + action_str + '\', Begin folder: \'' + str(begin_folder) + '\'')
        print('----------------------------------------------------------')

        while(True):
            print('From : ', end='')
            begin_img = input()
            if begin_img == 'Q' or begin_img == 'q' or begin_img == 'exit':
                break
            print('To   : ', end='')
            end_img = input()
            if end_img == 'Q' or end_img == 'q' or end_img == 'exit':
                continue
            elif (int(end_img) < int(begin_img)):
                print('type error')
                continue

            folder_name = action_label + '_{:04d}'.format(begin_folder)
            if not os.path.exists(root_path + action_label + '/' + folder_name + '/'):
                os.mkdir(root_path + action_label + '/' + folder_name + '/')
                for i in range(int(begin_img), int(end_img)+1):
                    copyfile(root_path + 'Raw_data/' + person + '/img_{:05d}.jpg'.format(i), 
                            root_path + action_label + '/' + folder_name + '/img_{:05d}.jpg'.format(i-int(begin_img)+1))
                print(folder_name ,'has been created. Copy from', begin_img, 'to', end_img, '({:d})'.format( int(end_img)-int(begin_img) + 1 ))
            else:
                print('Folder existed !!!!!!!!!!!!!!!!!!!!!!')

            begin_folder += 1
            
            '''
            folder_name = input()
            if folder_name == 'Q' or folder_name == 'q' or folder_name == 'exit':
                break
            folder_name = action_label + '_{:04d}'.format(int(folder_name))
            if not os.path.exists('D:/Dataset/Action/my_dataset/crop/' + action_label + '/' + folder_name + '/'):
                os.mkdir('D:/Dataset/Action/my_dataset/crop/' + action_label + '/' + folder_name + '/')
                print(folder_name ,'has been created')
            else:
                print('folder existed')
            '''