import os

# file_arr = ['1', '2']

#for arr in file_arr:
file = os.listdir('6')
file_len = len(file)
print(file_len)

cou = 1
for dirPath, dirNames, fileNames in os.walk('6'):
    if (len(dirPath)==8):
        new_path = dirPath.split('_')[0] + '_' + '{:04d}'.format(cou)
        print(dirPath + ' -> ' + new_path)
        #print('dirNames : ', dirNames)
        os.rename(dirPath, new_path)
        cou += 1
    '''
    for f in dirNames:
        dir_path = os.path.join(dirPath, f)  # 1\1_0001
        #print(dir_path)
        new_dir_path = dir_path.replace('1_', '2_')
        print(new_dir_path)
        os.rename(dir_path, new_dir_path)
    '''
 