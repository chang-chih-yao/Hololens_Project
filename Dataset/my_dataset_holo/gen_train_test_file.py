import os

#file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3_start', 'crop\\3_end', 'crop\\4_start', 'crop\\4_end', 'crop\\5_start', 'crop\\5_end', 'crop\\6', 'crop\\7', 'crop\\8', 'crop\\9', 'crop\\10', 'crop\\11']   # 15 classes

#file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3_start', 'crop\\3_end', 'crop\\4_start', 'crop\\4_end', 'crop\\5_start', 'crop\\5_end', 'crop\\6', 'crop\\7', 'crop\\9', 'crop\\10']   # 13 classes

file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4_start', 'crop\\4_end', 'crop\\5', 'crop\\6', 'crop\\7', 'crop\\9', 'crop\\10']   # 11 classes

person_index_1 = [[1,136],[137,150],[151,160],[161,162],[163,173],[174,179],[180,188],[178,196],[197,204],[205,217],[218,245],[246,264],[265,276]]

person_index_2 = [[1,83],[84,97],[98,112],[113,125],[126,144],[145,154],[155,169],[170,184],[185,199],[200,214],[215,235],[236,263],[264,281]]

person_index_3 = [[1,76],[77,90],[91,105],[106,118],[119,137],[138,147],[148,162],[163,177],[178,192],[193,207],[208,228],[229,256],[257,274]]

person_index_4 = [[],[1,15],[16,29],[30,37],[38,62],[63,80],[81,95],[96,110],[111,125],[126,139],[140,160],[161,180],[]]

person_index_5 = [[],[1,15],[16,31],[32,46],[47,69],[64,78],[79,93],[94,108],[109,122],[123,137],[138,157],[158,177],[]]

person_index_6 = [[],[1,15],[16,30],[31,42],[43,56],[57,71],[72,86],[87,101],[102,115],[116,130],[131,150],[151,170],[]]

person_index_7 = [[],[1,14],[15,27],[28,39],[40,54],[55,71],[72,85],[86,100],[101,115],[116,131],[132,151],[152,172],[173,191]]

def MOD_X(MOD_NUM=3):

    print('MOD_' + str(MOD_NUM))

    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    #size_1 = 34
    #size_2 = 38

    label_arr = [' 1\n', ' 2\n', ' 3\n', ' 4\n', ' 5\n', ' 6\n', ' 7\n', ' 8\n', ' 9\n', ' 10\n', ' 11\n', ' 12\n', ' 13\n', ' 14\n', ' 15\n']

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames)
            s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_holo\\' + dirPath + ' ' + str(len(fileNames)) + label_arr[arr]
            if len(fileNames) != 0:
                if cou % MOD_NUM == 0:
                    f_test.write(s)
                    cou += 1
                else:
                    f_train.write(s)
                    cou += 1


    f_train.close()
    f_test.close()

def for_7_class(MOD_NUM=3):
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']
    label_arr = [' 1\n', ' 2\n', ' 3\n', ' 4\n', ' 5\n', ' 6\n', ' 7\n']

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames)
            s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_holo\\' + dirPath + ' ' + str(len(fileNames)) + label_arr[arr]
            if len(fileNames) != 0:
                if cou % MOD_NUM == 0:
                    f_test.write(s)
                    cou += 1
                else:
                    f_train.write(s)
                    cou += 1


    f_train.close()
    f_test.close()

def for_cross_val_7_class():
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    file_arr = ['crop\\1', 'crop\\2_start', 'crop\\2_end', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']
    label_arr = [' 1\n', ' 2\n', ' 3\n', ' 4\n', ' 5\n', ' 6\n', ' 7\n']

    #test_person = [1,4,7]   # 13個人當中選3個來當testing data
    test_person = [11]   # 13個人當中選1個來當testing data

    print('test_person = ', end='')
    print(test_person)

    for arr in range(len(file_arr)):
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames)
            s = 'D:\\Code\\Hololens_Project\\Dataset\\my_dataset_holo\\' + dirPath + ' ' + str(len(fileNames)) + label_arr[arr]
            if len(dirPath) == 13 or len(dirPath) == 17 or len(dirPath) == 15:
                if len(dirPath) == 13:
                    my_index = int(dirPath.split('_')[-1])
                elif len(dirPath) == 17 or len(dirPath) == 15:
                    my_index = int(dirPath.split('\\')[-1])
                for_loop_cou = 0
                for i in range(len(test_person)):
                    if dirPath.split('\\')[1] == '1' and person_index_1[test_person[i]][0] <= my_index and my_index <= person_index_1[test_person[i]][1]:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '2_start' and person_index_2[test_person[i]][0] <= my_index and my_index <= person_index_2[test_person[i]][1]:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '2_end' and person_index_3[test_person[i]][0] <= my_index and my_index <= person_index_3[test_person[i]][1]:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '3' and person_index_4[test_person[i]][0] <= my_index and my_index <= person_index_4[test_person[i]][1]:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '4' and person_index_5[test_person[i]][0] <= my_index and my_index <= person_index_5[test_person[i]][1]:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '5' and person_index_6[test_person[i]][0] <= my_index and my_index <= person_index_6[test_person[i]][1]:
                        f_test.write(s)
                    elif dirPath.split('\\')[1] == '6' and person_index_7[test_person[i]][0] <= my_index and my_index <= person_index_7[test_person[i]][1]:
                        f_test.write(s)
                    else:
                        for_loop_cou += 1
                if for_loop_cou == len(test_person):
                    f_train.write(s)

    f_train.close()
    f_test.close()


if __name__ == "__main__":
    MOD_X(MOD_NUM=4)
    #for_7_class(MOD_NUM=3)
    #for_cross_val_7_class()
    