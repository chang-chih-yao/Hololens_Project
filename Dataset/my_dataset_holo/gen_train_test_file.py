import os

def MOD_3():
    f_train = open('my_train.txt', 'w')
    f_test = open('my_test.txt', 'w')

    #size_1 = 34
    #size_2 = 38

    file_arr = ['crop\\1', 'crop\\2', 'crop\\3', 'crop\\4', 'crop\\5', 'crop\\6']
    label_arr = [' 1\n', ' 2\n', ' 3\n', ' 4\n', ' 5\n', ' 6\n']

    for arr in range(len(file_arr)):
        cou = 0
        for dirPath, dirNames, fileNames in os.walk(file_arr[arr]):
            #print(dirPath, dirNames, fileNames)
            s = 'D:\\Dataset\\Action\\my_dataset\\' + dirPath + ' ' + str(len(fileNames)) + label_arr[arr]
            if len(fileNames) != 0:
                if cou % 3 == 0:
                    f_test.write(s)
                    cou += 1
                else:
                    f_train.write(s)
                    cou += 1


    f_train.close()
    f_test.close()



if __name__ == "__main__":
    MOD_3()