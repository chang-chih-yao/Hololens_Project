import os

def MOD_3_21_class():
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    file_arr = ['crop\\1', 'crop\\2', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']
    
    label = -3

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''

            if len(dirPath) == 8:
                label += 1

            if len(dirPath) == 13:
            
                if dirPath.split('\\')[1] == '1':
                    s = 'D:\\Dataset\\Action\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' 1\n'
                else:
                    #print(label)
                    s = 'D:\\Dataset\\Action\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                
                if len(fileNames) != 0:
                    if cou % 3 == 0:
                        f_test.write(s)
                        cou += 1
                    else:
                        f_train.write(s)
                        cou += 1
    f_train.close()
    f_test.close()

def MOD_3_6_class():
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    file_arr = ['crop\\1', 'crop\\2', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''
            if len(dirPath) == 13:
                s = 'D:\\Dataset\\Action\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + dirPath.split('\\')[1] + '\n'
                if len(fileNames) != 0:
                    if cou % 3 == 0:
                        f_test.write(s)
                        cou += 1
                    else:
                        f_train.write(s)
                        cou += 1
    f_train.close()
    f_test.close()


def cross_val_6_class():
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    file_arr = ['crop\\1', 'crop\\2', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''

            if len(dirPath) == 13:
                s = 'D:\\Dataset\\Action\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + dirPath.split('\\')[1] + '\n'

                if len(fileNames) != 0:
                    if dirPath.split('\\')[1] == '1' and int(dirPath.split('\\')[-1].replace('\n', '')) >= 187:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '2' and int(dirPath.split('\\')[-1].replace('\n', '')) >= 272:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '3' and int(dirPath.split('\\')[-1].replace('\n', '')) >= 300:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '4' and int(dirPath.split('\\')[-1].replace('\n', '')) >= 290:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '5' and int(dirPath.split('\\')[-1].replace('\n', '')) >= 260:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '6' and int(dirPath.split('\\')[-1].replace('\n', '')) >= 297:
                        f_test.write(s)
                    else:
                        f_train.write(s)
    f_train.close()
    f_test.close()


def main():
    #MOD_3_6_class()
    #MOD_3_21_class()
    cross_val_6_class()
    


if __name__ == '__main__':
    main()

