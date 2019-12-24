import os

# file_arr = ['1', '2']

# file = os.listdir('7')
# file_len = len(file)
# print(file_len)

now_folder_num = 1

cou = 1
for dirPath, dirNames, fileNames in os.walk('10'):
    # if (len(dirPath)==8):
    #     print(dirPath)
    '''
    if (len(dirPath)==8):
        new_path = dirPath.split('_')[0] + '_' + '{:04d}'.format(cou)
        print(dirPath + ' -> ' + new_path)
        #print('dirNames : ', dirNames)
        #os.rename(dirPath, new_path)
        cou += 1
    '''
    
    for f in fileNames:
        dir_path = os.path.join(dirPath, f)  # 1\1_0001
        dir_path = dir_path.replace('\\', '/')
        #print(dir_path)
        
        if now_folder_num != int(dir_path.split('/')[1].split('_')[1]):
            now_folder_num = int(dir_path.split('/')[1].split('_')[1])
            cou = 1
        img_name = 'img_{:05d}.jpg'.format(cou)
        new_dir_path = dir_path.split('i')[0] + img_name
        # new_dir_path = dir_path.replace('1_', '2_')
        print(dir_path + ' -> ' + new_dir_path)
        os.rename(dir_path, new_dir_path)
        cou += 1
    
 