import os

file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

person_index_1 = [[1,12], [13,30], [31,41], [42,52], [53,63], [64,79], [80,89], [90,98], [99,110], [111,118], [119,130], [131,143], [144,154], [155,175], [176,186], [187,208], [209,219], [220,236], [237,262]]

person_index_2 = [[1,20],[21,40],[41,49],[50,69],[70,89],[90,106],[107,126],[127,145],[146,164],[165,186],[187,206],[207,224],[225,231],[232,251],[252,271],[272,285],[286,305],[306,326],[327,349]]

person_index_3 = [[1,20],[21,39],[40,59],[60,79],[80,99],[100,119],[120,138],[139,158],[159,176],[177,198],[199,219],[220,239],[240,259],[260,279],[280,299],[300,320],[321,359],[360,404],[405,436]]

person_index_4 = [[1,20],[21,40],[41,60],[61,80],[81,100],[101,119],[],[120,139],[140,159],[160,179],[180,201],[202,221],[222,249],[250,269],[270,289],[290,309],[310,329],[330,352],[353,372]]

person_index_5 = [[1,20],[],[],[21,43],[44,61],[62,80],[81,100],[101,120],[121,140],[141,160],[161,180],[181,200],[201,219],[220,239],[240,259],[260,280],[281,300],[301,319],[320,339]]

person_index_6 = [[1,20],[21,40],[41,61],[62,81],[82,101],[102,121],[122,142],[143,162],[163,182],[183,201],[202,221],[222,237],[238,256],[257,276],[277,296],[297,316],[317,339],[340,360],[361,382]]

def for_6_class(MOD_NUM=3):
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    #file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    label = 0

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''
            if len(dirPath.split('\\')) == 2:
                label += 1
            if len(dirPath.split('\\')) == 4:
                s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                if len(fileNames) != 0:
                    if cou % MOD_NUM == 0:
                        f_test.write(s)
                        cou += 1
                    else:
                        f_train.write(s)
                        cou += 1
    f_train.close()
    f_test.close()

def for_21_class(MOD_NUM=3):
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    #file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']
    
    label = -3

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath, len(dirPath.split('\\')))
            s = ''
            
            if len(dirPath.split('\\')) == 3:
                label += 1

            if len(dirPath.split('\\')) == 4:
                if dirPath.split('\\')[1] == '1':
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' 1\n'
                else:
                    #print(label)
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                
                if len(fileNames) != 0:
                    if cou % MOD_NUM == 0:
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

    #file_arr = ['crop\\1', 'crop\\2', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    #test_person = [0,7,13,15,17]   # 19個人當中選5個來當testing data
    test_person = [18]               # 19個人當中選1個來當testing data

    print('cross_val_6_class, test_person = ', end='')
    print(test_person)

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''

            if len(dirPath) == 13:
                s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + dirPath.split('\\')[1] + '\n'

                if len(fileNames) != 0:
                    my_index = int(dirPath.split('\\')[-1].replace('\n', ''))
                    for_loop_cou = 0
                    for i in range(len(test_person)):
                        if dirPath.split('\\')[1] == '1' and person_index_1[test_person[i]][0] <= my_index and my_index <= person_index_1[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '2' and person_index_2[test_person[i]][0] <= my_index and my_index <= person_index_2[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '3' and person_index_3[test_person[i]][0] <= my_index and my_index <= person_index_3[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '4' and person_index_4[test_person[i]][0] <= my_index and my_index <= person_index_4[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '5' and person_index_5[test_person[i]][0] <= my_index and my_index <= person_index_5[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '6' and person_index_6[test_person[i]][0] <= my_index and my_index <= person_index_6[test_person[i]][1]:
                            f_test.write(s)
                        else:
                            for_loop_cou += 1
                    if for_loop_cou == len(test_person):
                        f_train.write(s)
    f_train.close()
    f_test.close()


def cross_val_21_class():
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    #file_arr = ['crop\\1', 'crop\\2', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    #test_person = [0,7,13,15,17]   # 19個人當中選5個來當testing data
    test_person = [4]               # 19個人當中選1個來當testing data

    print('cross_val_21_class, test_person = ', end='')
    print(test_person)

    label = -3

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''

            if len(dirPath) == 8:
                label += 1

            if len(dirPath) == 13:
                if dirPath.split('\\')[1] == '1':
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' 1\n'
                else:
                    #print(label)
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                
                if len(fileNames) != 0:
                    my_index = int(dirPath.split('\\')[-1].replace('\n', ''))
                    for_loop_cou = 0
                    for i in range(len(test_person)):
                        if dirPath.split('\\')[1] == '1' and person_index_1[test_person[i]][0] <= my_index and my_index <= person_index_1[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '2' and person_index_2[test_person[i]][0] <= my_index and my_index <= person_index_2[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '3' and person_index_3[test_person[i]][0] <= my_index and my_index <= person_index_3[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '4' and person_index_4[test_person[i]][0] <= my_index and my_index <= person_index_4[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '5' and person_index_5[test_person[i]][0] <= my_index and my_index <= person_index_5[test_person[i]][1]:
                            f_test.write(s)
                        elif dirPath.split('\\')[1] == '6' and person_index_6[test_person[i]][0] <= my_index and my_index <= person_index_6[test_person[i]][1]:
                            f_test.write(s)
                        else:
                            for_loop_cou += 1
                    if for_loop_cou == len(test_person):
                        f_train.write(s)
    f_train.close()
    f_test.close()

def one_cam_testing_data(cam_Num=0):
    print(cam_Num, 'camera test list')
    f_test = open('one_cam_test.txt', 'w')

    #file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    label = 0

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''
            if len(dirPath.split('\\')) == 2:
                label += 1
            if len(dirPath.split('\\')) == 4:
                if dirPath.split('\\')[2] == str(cam_Num):
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                    f_test.write(s)
    f_test.close()

def cross_val_cam(cam_Num=0):
    print(cam_Num, 'camera cross validation list')
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')
    
    #file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']

    label = 0

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''
            if len(dirPath.split('\\')) == 2:
                label += 1
            if len(dirPath.split('\\')) == 4:
                if dirPath.split('\\')[2] == str(cam_Num):
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                    f_test.write(s)
                else:
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                    f_train.write(s)
    f_train.close()
    f_test.close()

def cross_view(train_cam=0, test_cam=1):
    print('(%d, %d)' % (train_cam, test_cam))
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    label = 0

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            #print(dirPath)
            s = ''
            if len(dirPath.split('\\')) == 2:
                label += 1
            if len(dirPath.split('\\')) == 4:
                if dirPath.split('\\')[2] == str(test_cam):
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                    f_test.write(s)
                elif dirPath.split('\\')[2] == str(train_cam):
                    s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_4_cam\\' + dirPath + ' ' + str(len(fileNames)) + ' ' + str(label) + '\n'
                    f_train.write(s)
    f_train.close()
    f_test.close()

def main():
    #for_6_class(MOD_NUM=4)
    #for_21_class(MOD_NUM=4)

    #cross_val_6_class()
    #cross_val_21_class()

    #one_cam_testing_data(3)

    #cross_val_cam(3)

    cross_view(1, 2)


if __name__ == '__main__':
    main()

