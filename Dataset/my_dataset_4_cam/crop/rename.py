import os

file_arr = ['1', '2']

#for arr in file_arr:
file = os.listdir('1')
file_len = len(file)
print(file_len)
for dirPath, dirNames, fileNames in os.walk('1'):
    print('dirPath : ', dirPath)
    print('dirNames : ', dirNames)
    
    cou = 1
    '''
    for f in dirNames:
        dir_path = os.path.join(dirPath, f)  # 1\1_0001
        #print(dir_path)
        new_dir_path = dir_path.replace('1_', '2_')
        print(new_dir_path)
        os.rename(dir_path, new_dir_path)
    '''
 