import os

file_arr = ['6']

for arr in file_arr:
    cou = 1
    for dirPath, dirNames, fileNames in os.walk(arr):
        if (len(dirPath)==8):
            print('dirPath : ', dirPath, end='')
            print(' ', end='')
            #print('dirNames : ', dirNames)
            
            
            num = int(dirPath.split('_')[-1])
            new_dir = dirPath.split('_')[0] + '_{:04d}'.format(cou)
            print(new_dir)
            cou += 1
            os.rename(dirPath, new_dir)

