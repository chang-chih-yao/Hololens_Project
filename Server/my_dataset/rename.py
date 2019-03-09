import os

file_arr = ['1', '2']

for arr in file_arr:
    for dirPath, dirNames, fileNames in os.walk(arr):
        print('dirPath : ', dirPath)
        print('dirNames : ', dirNames)
        
        cou = 1
        '''
        for f in dirNames:
            dir_path = os.path.join(dirPath, f)  # 1\1_0001
            #print(dir_path)
            new_dir_path = dir_path.replace('2_', '1_')
            print(new_dir_path)
            os.rename(dir_path, new_dir_path)
        '''

        for f in fileNames:
            img_dir = os.path.join(dirPath, f)
            #print(img_dir)
            new_img_dir = img_dir.replace(f, 'img_{:05d}.jpg'.format(cou))
            os.rename(img_dir, new_img_dir)
            cou += 1
