import os

file_arr = ['1']

#for arr in file_arr:
file = os.listdir('1')
file_len = len(file)
print(file_len)
for dirPath, dirNames, fileNames in os.walk('1'):
    # print('dirPath : ', dirPath)
    # print('dirNames : ', dirNames)

    if len(dirPath) == 8:
        folder_num = int(dirPath.split('\\')[-1])
        if folder_num > 208:
            # print('dirPath : ', dirPath)
            # print('dirNames : ', dirNames)
            # print(folder_num, end='')
            # print(' -> ', end='')
            # print(folder_num-4)
            new_folder_num = '1\\' + dirPath.split('\\')[1] + '\\' + '{:04d}'.format(folder_num-4)
            #print(new_folder_num)
            os.rename(dirPath, new_folder_num)
    
    '''
    for f in dirNames:
        dir_path = os.path.join(dirPath, f)  # 1\1_0001
        #print(dir_path)
        new_dir_path = dir_path.replace('1_', '2_')
        print(new_dir_path)
        os.rename(dir_path, new_dir_path)
    '''
 