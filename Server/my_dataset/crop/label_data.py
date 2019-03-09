import os
from shutil import copyfile

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
        
        folder_num = len(os.listdir('D:/Dataset/Action/my_dataset/crop/' + action_label + '/'))
        begin_folder = folder_num + 1
        print('Begin folder         :', begin_folder)

        print('Person: \'' + person + '\', Action: \'' + action_label + '\', Begin folder: \'' + str(begin_folder) + '\'')
        print('--------------------------------------------------')

        while(True):
            print('From : ', end='')
            begin_img = input()
            if begin_img == 'Q' or begin_img == 'q' or begin_img == 'exit':
                break
            print('To   : ', end='')
            end_img = input()

            folder_name = action_label + '_{:04d}'.format(begin_folder)
            if not os.path.exists('D:/Dataset/Action/my_dataset/crop/' + action_label + '/' + folder_name + '/'):
                os.mkdir('D:/Dataset/Action/my_dataset/crop/' + action_label + '/' + folder_name + '/')
                for i in range(int(begin_img), int(end_img)+1):
                    copyfile('D:/Dataset/Action/my_dataset/Raw_data/' + person + '/img_{:05d}.jpg'.format(i), 
                            'D:/Dataset/Action/my_dataset/crop/' + action_label + '/' + folder_name + '/img_{:05d}.jpg'.format(i-int(begin_img)+1))
                print(folder_name ,'has been created. Copy from', begin_img, 'to', end_img)
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