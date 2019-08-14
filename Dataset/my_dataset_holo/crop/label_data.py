import os
from shutil import copyfile

while(True):
    print('which action : ', end='')
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

    folder_num = len(os.listdir(action_label + '_start/'))
    start_cou = folder_num + 1
    folder_num = len(os.listdir(action_label + '_end/'))
    end_cou = folder_num + 1
    print('Begin folder :  start ' + str(start_cou) + ' end ' + str(end_cou))

    print('----------------------------------------------------------')

    while(True):
        print('In ' + action_label + '_{:04d}/  '.format(start_cou), end='')
        print('Input index : ', end='')
        index = input()
        if index == 'q':
            break

        if index != '0':
            if not os.path.exists(action_label + '_start/' + '{:04d}'.format(start_cou)):
                os.mkdir(action_label + '_start/' + '{:04d}/'.format(start_cou))
            if not os.path.exists(action_label + '_end/' + '{:04d}'.format(end_cou)):
                os.mkdir(action_label + '_end/' + '{:04d}/'.format(end_cou))
            
            for s in range(int(index)):
                copyfile(action_label + '/' + action_label + '_{:04d}/'.format(start_cou) + 'img_{:05d}.jpg.'.format(s+1), 
                        action_label + '_start/' + '{:04d}/'.format(start_cou) + 'img_{:05d}.jpg'.format(s+1))
            for e in range(int(index), len(os.listdir(action_label + '/' + action_label + '_{:04d}/'.format(start_cou)))):
                copyfile(action_label + '/' + action_label + '_{:04d}/'.format(start_cou) + 'img_{:05d}.jpg.'.format(e+1), 
                        action_label + '_end/' + '{:04d}/'.format(end_cou) + 'img_{:05d}.jpg'.format(e-int(index)+1))
            
            start_cou += 1
            end_cou += 1
        else:   # 沒有結束動作
            if not os.path.exists(action_label + '_start/' + '{:04d}'.format(start_cou)):
                os.mkdir(action_label + '_start/' + '{:04d}/'.format(start_cou))

            for s in range(len(os.listdir(action_label + '/' + action_label + '_{:04d}/'.format(start_cou)))):
                copyfile(action_label + '/' + action_label + '_{:04d}/'.format(start_cou) + 'img_{:05d}.jpg.'.format(s+1), 
                        action_label + '_start/' + '{:04d}/'.format(start_cou) + 'img_{:05d}.jpg'.format(s+1))
            
            start_cou += 1

            